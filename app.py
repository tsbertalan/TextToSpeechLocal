# https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html
# Let's adapt the above tut1 to make a little tk app that says things you put in a text box, after you click a "speak" button.

import tkinter as tk, re
from tkinter import ttk
import torch, torchaudio, numpy as np
import sounddevice as sd, time, os, datetime


# Add to env from .env file in current directory.
HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(HERE, '.env')):
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, '.env'))

import tempfile
TEMPDIR = tempfile.gettempdir()

import joblib
joblib_cache = joblib.Memory(os.path.join(TEMPDIR, 'joblib'), verbose=0)

@joblib_cache.cache(ignore=['oai_client'])  # Save some bux
def openai_tts(text, oai_client, model='tts-1', voice='alloy'):
    _speech_tempfile_path = os.path.join(TEMPDIR, "tts.wav")
    response = oai_client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    # Suppress the DeprecationWarning from stream_to_file. https://community.openai.com/t/tts-does-not-work-curl-python/609455/2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") 
        response.stream_to_file(_speech_tempfile_path)
    wave, rate = torchaudio.load(_speech_tempfile_path)
    return wave[0], rate


def get_wave_duration(array, sample_rate):
    """Given a sound wave, return its duration in seconds."""
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    return array.shape[0] / sample_rate


def play_wave(array, sample_rate=22050, extra_delay=0, do_sleep=False):
    """Play a sound wave."""
    if not isinstance(sample_rate, int):
        sample_rate = int(sample_rate)
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    sd.play(array, sample_rate, blocking=True)
    if do_sleep:
        time.sleep(get_wave_duration(array, sample_rate))
        if extra_delay > 0:
            time.sleep(extra_delay)


class SpeakerModels:

    def __init__(self, message_callback=lambda s: None, use_whisper=True, speaker_silero=None):
        """Contains the tokenizer, spectrogrammer (tacotron2), and vocoder (waveglow)."""
        torch.random.manual_seed(0)
        use_tacotron = False
        device = (
            "cuda" if torch.cuda.is_available()
            # and use_tacotron
            else "cpu"
        )
        self.device = device


        if use_whisper:
            from openai import OpenAI
            self.oai_client = OpenAI()
            self.oai_model = "tts-1"
            self.oai_voice = "alloy"
            
            self._speech_tempfile_path = os.path.join(TEMPDIR, "tts.wav")

            def _get_wave(text, message_callback=lambda s: None):
                """Given a string, return a sound wave."""
                message_callback(f"OAI-TTS ({self.oai_model}:{self.oai_voice}): {repr(text)}")
                wave, rate = openai_tts(text, self.oai_client, model=self.oai_model, voice=self.oai_voice)
                self.sample_rate = rate
                return wave
            self.get_wave = _get_wave

        elif not use_tacotron:
            use_audiocraft = False  # This doesn't work. Not TTS; for sound effects instead, it seems.
            if use_audiocraft:
                from audiocraft.models import AudioGen
                # from audiocraft.data.audio import audio_write

                model = AudioGen.get_pretrained('facebook/audiogen-medium')
                # model.set_generation_params(duration=5)  # generate 5 seconds.
                self.sample_rate = 48000
                # descriptions = ['dog barking', 'sirene of an emergency vehicle', 'footsteps in a corridor']
                # wav = model.generate(descriptions)  # generates 3 samples.

                # for idx, one_wav in enumerate(wav):
                #     # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
                #     audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

                def _get_wave(text, message_callback=lambda s: None):
                    """Given a string, return a sound wave for a narration."""
                    text = "A narrator speaking the following text: \"" + text + "\""  # Add a little intro because this is a general "sound foundation model", not a simple TTS.
                    message_callback(f"Prompt: {repr(text)}", notime=True, tag='text')
                    descriptions = [text]
                    wavs = model.generate(descriptions)
                    wavs = torch.squeeze(wavs)
                    message_callback(f"Generated {len(wavs)} wavs.", tag='text')
                    return wavs
                
                self.get_wave = _get_wave


            else:

                torch._C._jit_set_profiling_mode(False)
                
                language = 'en'
                model_id = 'v3_en'
                if speaker_silero is None:
                    speaker_silero = 'en_13'   # I like 0, 10, 13, 14, 15, 25
                speaker = speaker_silero  # en_0, en_1, ..., en_117, random
                self.sample_rate = 48000
                self.silero_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                    model='silero_tts',
                                                    language=language,
                                                    speaker=model_id)
                
                self.silero_model.to(device)  # gpu or cpu

                message_callback(f"loaded silero_tts (model_id={model_id}, speaker={speaker}) on {device}.")

                def _get_wave(text, message_callback=lambda s: None):
                    """Given a string, return a sound wave."""
                    intshape = lambda tensor: tuple(int(i) for i in tensor.shape)
                    with torch.no_grad():
                        audio = self.silero_model.apply_tts(text=text, speaker=speaker, sample_rate=self.sample_rate)
                    return audio
                
                self.get_wave = _get_wave

        else:
            self.sample_rate = 22050
            bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
            message_callback(f"loaded tacotron2 on {device}.")

            self.processor = bundle.get_text_processor()
            self.tacotron2 = bundle.get_tacotron2().to(device)

            vocoder_method = 'waveglow'

            if vocoder_method == 'wavernn':
                unchunked_vocoder = bundle.get_vocoder().to(device)
                def unchunked_vocode(specs, lengths):
                    return unchunked_vocoder(specs)[0]
                self.vocode = unchunked_vocode

            elif vocoder_method == 'melgan':
                melgan = torch.hub.load('seungwonpark/melgan', 'melgan')
                melgan.eval()
                melgan = melgan.to(device)
                message_callback(f"loaded Mel-GAN on {device}.")

                def vocode(spec, unused_lengths):
                    with torch.no_grad():
                        waveform = melgan.inference(spec)
                    return waveform.reshape(spec.shape[0], -1)
                self.vocode = vocode

            else:
                assert vocoder_method == 'waveglow'
                # Workaround to load model mapped on GPU
                # https://stackoverflow.com/a/61840832
                message_callback(f"loaded text processor.")
                waveglow = torch.hub.load(
                    "NVIDIA/DeepLearningExamples:torchhub",
                    "nvidia_waveglow",
                    model_math="fp32",
                    pretrained=False,
                )
                checkpoint = torch.hub.load_state_dict_from_url(
                    "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",  # noqa: E501
                    progress=True,
                    map_location=device,
                )
                state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

                waveglow.load_state_dict(state_dict)
                waveglow = waveglow.remove_weightnorm(waveglow)
                waveglow = waveglow.to(device)
                waveglow.eval()
                self.waveglow = waveglow
                message_callback(f"loaded waveglow on {device}.")

                def vocode(spec, unused_lengths):
                    return self.waveglow.infer(spec)
                self.vocode = vocode

            def _get_wave(text, message_callback=lambda s: None):
                """Given a string, return a sound wave."""
                with torch.no_grad():
                    with torch.inference_mode():
                        tokens, lengths = self.processor(text)
                        intshape = lambda tensor: tuple(int(i) for i in tensor.shape)
                        message_callback(f"Preprocessed text of length {len(text)}, making tokens of shape {intshape(tokens)}.")
                        tokens = tokens.to(self.device)
                        lengths = lengths.to(self.device)
                        
                        # message_callback(f"moved to {self.device} ...")
                        # message_callback(f"Lengths: {lengths}")
                        # print("lengths:", lengths)
                        spec, spec_lengths, _ = self.tacotron2.infer(tokens, lengths)
                        message_callback(f"Created spectrogram of shape {intshape(spec)}")

                        waveforms = self.vocode(spec, spec_lengths)
                        message_callback(f"Vocoder output has shape {intshape(waveforms)}")
                        waveform = waveforms[0].cpu().detach().numpy()
                        message_callback(f"created waveform of shape {intshape(waveform)}")
                return waveform

            self.get_wave = _get_wave

    def play_wave(self, array, sample_rate=None, extra_delay=0):
        if sample_rate is None:
            sample_rate = self.sample_rate
        play_wave(array, sample_rate, extra_delay)

    def play_text(self, text, **kw_player):
        """Given a string, play it as a sound wave."""
        wave = self.get_wave(text)
        self.play_wave(wave, **kw_player)



def breakup(long_text):
    """Break up long text basically into sentences."""

    # Look for periods, question marks, and exclamation marks followed by ... some things that go after sentences.
    # This is a very naive approach, but it's good enough for now.
    output = []
    i1 = 0
    i2 = i1 + 1
    # regexes that indicate an end of chunk.
    # Must occur at the *end* of the tested string.
    # enders = " \n\t\"'”’"
    enders = [
        re.compile(s) for s in [
            # Period, question mark, or exclamation mark, or quote, or apostrophe; followed by whitespace followed by end-of-string:
            r'(.|\n)*[.?!\'”’][ \n\t]+$',
            # Double newline followed by end-of-string:
            r'(.|\n)*\n\n$',
        ]
    ]
    # not_enders = ["Mr", ]
    # Regexes that will countermand findings from one of the above.
    not_enders = [
        re.compile(s) for s in [
            # Mr., Mrs., Ms., Dr., St., etc. followed by whitespace followed by end-of-string.
            r'(.|\n)*(Mr|Mrs|Ms|Dr|St)[.][ \n\t]+$',
            # A (possibly multi-digit) number followed by a period followed by whitespace followed by end-of-string.
            r'(.|\n)*\d+[.][ \n\t]+$',
        ]
    ]

    long_text = long_text.strip()
    if len(long_text) == 0:
        return []
    while i2 < len(long_text):
        chunk = long_text[i1:i2]
        if any(r.match(chunk) for r in enders):
            if not any(r.match(chunk) for r in not_enders):
                chunk = chunk.strip()
                output.append(chunk)
                i1 = i2
                i2 = i1 + 1
            else:
                i2 += 1
        else:
            i2 += 1

    if i1 < len(long_text)-1:
        output.append(long_text[i1:].strip())

    # print('Broke up long text into', len(output), 'chunks:')
    # print(' \n'.join([f'>>{chunk}<<' for chunk in output]))

    return output


# # Now the "app" part:
# Let's put the speaker and its models on a separate thread, so the GUI doesn't freeze while it's speaking.
# There will be a long-lived worker thread that begins by loading up the models, then monitors a queue for text to speak.
# The GUI should include an output box that shows the length of the current queue.
# The GUI should watch for changes to the queue length, and update the output box accordingly.
# The button won't say "speak" now, but "Add to Queue".

# That's pretty good, but there's a big pause between sentences.
# Let's improve it by having *two* threads: one that loads the models and generates the audio as fast as it can, and another that speaks them.
# There will be *two* queues: one for text to speak, and one for audio to play. Both will be shown in the UI.

import queue, threading


class MessageTimer:

    def __init__(self, label: str, message_queue: queue.Queue, tag_padded_len: int=21, target_priority: int=0):
        self.label = label
        self.message_queue = message_queue
        self.tag_padded_len = tag_padded_len
        self.last_message_times = {}
        self.tic()
        self.target_priority = target_priority

    def tic(self, tag=None):
        self.last_message_times[tag] = time.time()

    def __call__(self, s, tag=None, notime=False, priority=None):
        if priority is None:
            priority = self.target_priority
        elapsed = time.time() - self.last_message_times.get(tag, self.last_message_times[None])
        self.last_message_times[tag] = time.time()
        if not notime:
            before_colon = f'{self.label} [{elapsed:.2f} s]'
        else:
            before_colon = self.label
        # Pad out before_colon to a fixed length.
        if len(before_colon) < self.tag_padded_len:
            before_colon += ' ' * (self.tag_padded_len - len(before_colon))

        tagged = f"{before_colon}: {s}"

        if priority >= self.target_priority:
            self.message_queue.put(tagged)



def text_to_audio_worker(text_queue, audio_queue, message_queue, command_queue, speaker_silero):
    # Make the speaker models.
    message_callback = MessageTimer("TTS MODEL", message_queue)
    message_callback("Loading models...", tag='loading', notime=True)

    speaker_models = SpeakerModels(message_callback=message_callback, speaker_silero=speaker_silero)

    try:
        message_queue.put(dict(sample_rate=speaker_models.sample_rate))
    except AttributeError:
        pass
    message_callback("Models loaded.", tag='loading')

    # Every time we see something in the queue, take it off and TTS it.
    while True:
        if not command_queue.empty():
            command = command_queue.get()
            if command == "time to die":
                break
        if not text_queue.empty():
            text = text_queue.get()
            message_callback.tic('tts')
            message_callback.tic('samples')
            try:
                wave = speaker_models.get_wave(text, message_callback=message_callback)
                if isinstance(wave, torch.Tensor):
                    wave = wave.cpu().detach().numpy()
                # Print the size padded to 16 characters.
                message_callback(f"Generated: {repr(text)}", tag='tts')
                message_callback(f"{wave.size:10} samples generated.", tag='samples', priority=-1)
                audio_queue.put((text, wave, speaker_models.sample_rate))
            except Exception as e:
                message_callback(f"Exception: {e}", tag='tts', priority=1)
        else:
            time.sleep(0.05)


def audio_speaking_worker(play_queue, message_queue, command_queue):
    # Every time we see something in the queue, take it off and speak it.
    message_callback = MessageTimer("SPEAKER", message_queue)
    samp_rate = 22050
    while True:
        if not command_queue.empty():
            command = command_queue.get()
            if str(command) == "time to die":
                break
            else:
                if isinstance(command, dict):
                    if 'sample_rate' in command:
                        samp_rate = command['sample_rate']
        if not play_queue.empty():
            info = play_queue.get()
            wave = info['audio']
            text = info['text']
            if 'sample_rate' in info:
                samp_rate = info['sample_rate']
            n_samp = np.asarray(wave).size
            message_callback.tic('speaking')
            try:
                message_callback(f"Speaking: {repr(text)}", notime=True, tag='text')
                info['playing'] = True
                play_wave(wave, sample_rate=samp_rate)
                info['playing'] = False
                info['queued'] = False
                message_callback(f"{n_samp:10} samples spoken at sample rate {samp_rate}.", tag='speaking', priority=-1)
            except Exception as e:
                message_callback(f"Exception: {e}", tag='speaking', priority=1)
        else:
            time.sleep(0.05)


class SpeakerApp(tk.Tk):

    def __init__(self, *args, speaker_silero=None, **kwargs):
        super().__init__(*args, **kwargs)    

        self.style = ttk.Style()
        from tkinter.constants import NORMAL, ACTIVE
        self.style_b0 = ttk.Style()
        self.style_b0.configure('button_0.TButton', foreground='black')
        self.style_b1 = ttk.Style()
        self.style_b1.configure('button_1.TButton', foreground='green')
        self.style_b2 = ttk.Style()
        self.style_b2.configure('button_2.TButton', foreground='blue')

        self.tts_queue = queue.Queue()
        self.audio_return_queue = queue.Queue()
        self.play_queue = queue.Queue()
        self.text_command_queue = queue.Queue()
        self.audio_command_queue = queue.Queue()
        self.worker_message_queue = queue.Queue()

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        HERE = os.path.dirname(os.path.abspath(__file__))
        self.tts_history_file_path = os.path.join(HERE, 'logs', f"TTS_{now}.md")
        self.tts_history_file = open(self.tts_history_file_path, 'w', encoding='utf8')

        self.sample_rate = 22050

        self.title("Speaker")
        self.geometry("1000x600")
        self.resizable(False, False)
        # Make our UI elements generally left-aligned.
        self.pack_propagate(0)

        # self.label = ttk.Label(self, text=f"Enter text to speak with silero voice {speaker_silero}.")
        self.label_input = ttk.Label(self, text=f"Enter text to queue for TTS:")
        self.label_input.pack()

        text_width = 360
        
        # Scrollable textbox for the input:
        self.input_textbox_frame = ttk.Frame(self)
        self.input_textbox_frame.pack()
        self.input_textbox_label = ttk.Label(self.input_textbox_frame, text="Text to speak:")
        self.input_textbox = tk.Text(self.input_textbox_frame, height=5, width=text_width)
        textbox_scrollbar = ttk.Scrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox["yscrollcommand"] = textbox_scrollbar.set
        textbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_textbox.pack()

        self.add_to_queue_button = ttk.Button(self, text="Add to Queue", command=self.add_to_tts_queue)
        self.add_to_queue_button.pack()

        self.queue_label = ttk.Label(self, text="Queue length: 0")
        self.queue_label.pack()

        self.play_queue_label = ttk.Label(self, text="Audio queue length: 0")
        self.play_queue_label.pack()

        # Make the log textbox not wrap, but instead have a horizontal scrollbar.
        self.message_textbox_frame = ttk.Frame(self, border=1, relief=tk.SUNKEN)
        self.message_textbox_frame.pack()
        self.message_textbox_label = ttk.Label(self.message_textbox_frame, text="Log Messages")
        self.message_textbox_label.pack(side=tk.TOP)
        self.message_textbox = tk.Text(self.message_textbox_frame, height=10, width=text_width, state=tk.DISABLED, wrap=tk.NONE)
        message_textbox_scrollbar = ttk.Scrollbar(self.message_textbox_frame, command=self.message_textbox.yview)
        self.message_textbox["yscrollcommand"] = message_textbox_scrollbar.set
        message_textbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        message_textbox_scrollbar_h = ttk.Scrollbar(self.message_textbox_frame, command=self.message_textbox.xview, orient=tk.HORIZONTAL)
        self.message_textbox["xscrollcommand"] = message_textbox_scrollbar_h.set
        message_textbox_scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        self.message_textbox.pack()

        # Make a horizontal scrollable box that we'll fill with buttons for each sentence.
        self.sentence_outer_frame = ttk.Frame(self, border=1, relief=tk.SUNKEN)
        self.sentence_outer_frame.pack(fill=tk.X)
        self.sentences_label = ttk.Label(self.sentence_outer_frame, text="Click an item to play starting from there in the list:")
        self.sentences_label.pack(side=tk.TOP)

        # self.sentence_buttons_label = ttk.Label(self.sentence_buttons_frame, text="Sentences:")
        # self.sentence_buttons_label.pack(side=tk.LEFT)

        self.sentence_canvas = tk.Canvas(self.sentence_outer_frame, height=100, width=1000, scrollregion=(0, 0, 10000, 100))
        self.sentence_inner_frame = ttk.Frame(self.sentence_canvas)

        self.sentence_hbar = ttk.Scrollbar(self.sentence_outer_frame, orient=tk.HORIZONTAL)
        self.sentence_hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.sentence_hbar.config(command=self.sentence_canvas.xview)

        self.sentence_canvas.config(xscrollcommand=self.sentence_hbar.set)
        self.sentence_canvas.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.sentence_inner_frame.bind(
            "<Configure>",
            lambda e: self.sentence_canvas.configure(
                scrollregion=self.sentence_canvas.bbox("all")
            )
        )
        self.sentence_canvas.create_window((0, 0), window=self.sentence_inner_frame, anchor="nw")

        # Add a few more buttons for some common tasks.
        self.extra_buttons_Frame = ttk.Frame(self, border=1, relief=tk.SUNKEN)
        self.extra_buttons_Frame.pack()
        self.clear_queue_button = ttk.Button(self.extra_buttons_Frame, text="Clear Play Queue", command=self.clear_play_queue)
        self.clear_queue_button.pack(side=tk.LEFT)
        self.clear_TTS_queue_button = ttk.Button(self.extra_buttons_Frame, text="Clear TTS Queue", command=self.clear_TTS_queue)
        self.clear_TTS_queue_button.pack(side=tk.LEFT)
        # ... I'll think of more later.

        self.results = {}
        
        # Start the worker threads.
        self.text_to_audio_worker = threading.Thread(target=text_to_audio_worker, args=(self.tts_queue, self.audio_return_queue, self.worker_message_queue, self.text_command_queue, speaker_silero))
        self.text_to_audio_worker.start()
        self.audio_speaking_worker = threading.Thread(target=audio_speaking_worker, args=(self.play_queue, self.worker_message_queue, self.audio_command_queue))
        self.audio_speaking_worker.start()

        # Ensure the queue_label gets updated regularly.
        self.update_ui_delay = 250  # ms
        self.after(self.update_ui_delay, self.update_ui)

    def add_sentence_button(self, text_id: int, text: str, audio: np.ndarray):
        """Add a button to the sentence_buttons canvas."""
        # The button will have width proportional to the audio size.
        # The button will have height 100.
        # The button text will be the sentence, but we'll let it run off the right side if it's too long.
        # On creation, buttons will have no special color. But when their corresponding audio is playing, they'll turn green.

        # Make the button with black text. 
        # Let the text be left-aligned in the box, so it only runs off on the right.
        button_width = max(20, int(audio.size * 5 / self.sample_rate))
        button = ttk.Button(self.sentence_inner_frame, text=text, width=button_width, style='button_0.TButton')
        button.pack(side=tk.LEFT)
        
        # Add the button to the dictionary.
        info = dict(
            text=text,
            text_id=text_id,
            button=button,
            audio=audio,
            playing=False,
            queued=True,
            sample_rate=self.sample_rate,
        )
        self.results[text_id] = info
        self.play_queue.put(info)
        # Add a callback to the button that will turn it green when its audio is playing.
        button.config(command=lambda: self.start_from_button(text_id))

    def clear_TTS_queue(self):
        while not self.tts_queue.empty():
            self.tts_queue.get()

    def clear_play_queue(self):
        while not self.play_queue.empty():
            existing = self.play_queue.get()
            existing['queued'] = False

    def start_from_button(self, text_id: int):
        """Start playing from here."""
        # Clear the play queue.
        self.clear_play_queue()
        
        # Add all audio from this text on to the play queue.
        for future_i in range(text_id, len(self.results)):
            item = self.results[future_i]
            item['queued'] = True
            self.play_queue.put(item)

    def check_for_tts_results(self):
        if not self.audio_return_queue.empty():
            text, audio, rate = self.audio_return_queue.get()
            self.sample_rate = rate
            text_id = len(self.results)
            self.add_sentence_button(text_id, text, audio)

    def update_ui(self):
        self.check_for_tts_results()

        self.queue_label["text"] = f"TTS queue length: {self.tts_queue.qsize()}"

        self.play_queue_label["text"] = f"Speaker queue length: {self.play_queue.qsize()}"

        # Set the color of all play buttons.
        for info in self.results.values():
            if info['playing']:
                info['button'].config(style="button_1.TButton")
            elif info['queued']:
                info['button'].config(style="button_2.TButton")
            else:
                info['button'].config(style="button_0.TButton")

        if not self.worker_message_queue.empty():
            message = self.worker_message_queue.get()
            if not isinstance(message, str):
                # The workers can pass data back to us also.
                self.handle_nonstring_message(message)
            else:
                self.add_log_message(message)
        self.after(self.update_ui_delay, self.update_ui)

    def add_log_message(self, message):
        """Append the message (after newline) to our message box existing text, and scroll to the bottom in the y direction."""
        self.message_textbox.configure(state=tk.NORMAL)
        self.message_textbox.insert(tk.END, message + "\n")
        # Scroll to the bottom, but keep the x position the same.
        x_pos = self.message_textbox.xview()[0]
        self.message_textbox.see(tk.END)
        self.message_textbox.xview_moveto(x_pos)
        self.message_textbox.configure(state=tk.DISABLED)

    def handle_nonstring_message(self, message):
        if isinstance(message, dict):
            if 'sample_rate' in message:
                self.audio_command_queue.put(message)
        
    def add_to_tts_queue(self):
        text = self.input_textbox.get("1.0", "end-1c")
        for chunk in breakup(text):
            self.tts_queue.put(chunk.replace('\n', ' '))
            self.tts_history_file.write(chunk + '\n')
            self.tts_history_file.flush()
        self.queue_label["text"] = f"Queue length: {self.tts_queue.qsize()}"

        # Clear the input box.
        self.input_textbox.delete("1.0", tk.END)

    def destroy(self):
        # Clicking the close button should just kill the whole process, threads be damned.
        # Drop a bunch of "time to die" messages into the command queue.
        for _ in range(10):
            self.text_command_queue.put("time to die")
            self.audio_command_queue.put("time to die")
        # Wait a little while for the threads to die.
        for _ in range(100):
            tts_alive = self.text_to_audio_worker.is_alive()
            speaker_alive = self.audio_speaking_worker.is_alive()
            if not self.text_to_audio_worker.is_alive() and not self.audio_speaking_worker.is_alive():
                break
            time.sleep(0.1)
        super().destroy()
        # If the threads are still alive, just kill everything.
        if tts_alive or speaker_alive:
            print(f"Threads still alive: TTS={tts_alive}, speaker={speaker_alive}")
            print("Threads still alive. Killing everything.")
            import os
            os._exit(0)

        self.tts_history_file.close()
        

if __name__ == "__main__":
    try:
        app = SpeakerApp()
        app.mainloop()
    except KeyboardInterrupt:
        print("Exiting app due to KeyboardInterrupt.")
        if 'app' in locals():
            app.destroy()
