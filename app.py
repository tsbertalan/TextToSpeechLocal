# https://pytorch.org/audio/stable/tutorials/tacotron2_pipeline_tutorial.html
# Let's adapt the above tut1 to make a little tk app that says things you put in a text box, after you click a "speak" button.

import tkinter as tk
import torch, torchaudio, numpy as np
import sounddevice as sd, time


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

    def __init__(self, message_callback=lambda s: None):
        """Contains the tokenizer, spectrogrammer (tacotron2), and vocoder (waveglow)."""
        torch.random.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_CHAR_LJSPEECH
        message_callback(f"loaded tacotron2 on {device}.")

        self.processor = bundle.get_text_processor()
        self.tacotron2 = bundle.get_tacotron2().to(device)

        vocoder_method = 'wavernn'


        if vocoder_method == 'wavernn':
            unchunked_vocoder = bundle.get_vocoder().to(device)
            
            def unchunked_vocode(specs, lengths):
                return unchunked_vocoder(specs)[0]

            self.vocode = unchunked_vocode
        else:
            assert vocoder_method == 'waveglow':
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
                waveform = self.waveglow.infer(spec)
                return waveform
            self.vocode = vocode

    def get_wave(self, text, message_callback=lambda s: None):
        """Given a string, return a sound wave."""
        with torch.no_grad():
            with torch.inference_mode():
                tokens, lengths = self.processor(text)
                intshape = lambda tensor: tuple(int(i) for i in tensor.shape)
                message_callback(f"Preprocessed text of length {len(text)}, making tokens of shape {intshape(tokens)}.")
                tokens = tokens.to(self.device)
                lengths = lengths.to(self.device)
                
                # message_callback(f"moved to {self.device} ...")
                spec, spec_lengths, _ = self.tacotron2.infer(tokens, lengths)
                message_callback(f"Created spectrogram of shape {intshape(spec)}")

                waveforms = self.vocode(spec, spec_lengths)
                waveform = waveforms[0].cpu().detach().numpy()
                message_callback(f"created waveform of shape {intshape(waveform)}")

        return waveform

    def play_wave(self, array, sample_rate=22050*1.2, extra_delay=0):
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
    enders = " \n\t\"'”’"
    not_enders = ["Mr", ]
    while i2 < len(long_text):
        if long_text[i2] in ".?!":
            if i2 == len(long_text) - 1 or long_text[i2+1] in enders:
                chunk = long_text[i1:i2]
                for c in enders:
                    if chunk.startswith(c):
                        chunk = chunk[1:]
                chunk = chunk.strip()
                ok_to_add = True
                for not_ender in not_enders:
                    if chunk.endswith(not_ender):
                        ok_to_add = False
                if ok_to_add:
                    output.append(chunk)
                    i1 = i2 + 1
                    i2 = i1 + 1
        i2 += 1

    if i1 < len(long_text):
        output.append(long_text[i1:])

    print('Broke up long text into', len(output), 'chunks:')
    print(' \n'.join([f'>>{chunk}<<' for chunk in output]))

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

    def __init__(self, label, message_queue, tag_padded_len=20):
        self.label = label
        self.message_queue = message_queue
        self.tag_padded_len = tag_padded_len
        self.last_message_times = {}
        self.tic()

    def tic(self, tag=None):
        self.last_message_times[tag] = time.time()

    def __call__(self, s, tag=None, notime=False):
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
        self.message_queue.put(tagged)



def text_to_audio_worker(text_queue, audio_queue, message_queue):
    # Make the speaker models.
    message_callback = MessageTimer("TTS MODELS", message_queue)
    message_callback("Loading models...", tag='loading', notime=True)
    speaker_models = SpeakerModels(message_callback=message_callback)
    message_callback("Models loaded.", tag='loading')

    # Every time we see something in the queue, take it off and TTS it.
    while True:
        if not text_queue.empty():
            text = text_queue.get()
            message_callback.tic()
            wave = speaker_models.get_wave(text, message_callback=message_callback)
            audio_queue.put(wave)
        else:
            time.sleep(0.05)


def audio_speaking_worker(audio_queue, message_queue):
    # Every time we see something in the queue, take it off and speak it.
    message_callback = MessageTimer("SPEAKER", message_queue)
    while True:
        if not audio_queue.empty():
            wave = audio_queue.get()
            n_samp = wave.size
            message_callback(f"{n_samp} samples to speak ...", tag='speaking', notime=True)
            play_wave(wave)  # blocking
            message_callback(f"{n_samp} samples spoken.", tag='speaking')
        else:
            time.sleep(0.05)


class SpeakerApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.worker_message_queue = queue.Queue()

        self.title("Speaker")
        self.geometry("1000x1000")
        self.resizable(False, False)

        self.label = tk.Label(self, text="Enter text to speak:")
        self.label.pack()

        text_width = 100
        
        # Scrollable textbox for the input:
        self.input_textbox_frame = tk.Frame(self)
        self.input_textbox_frame.pack()
        self.input_textbox_label = tk.Label(self.input_textbox_frame, text="Text to speak:")
        self.input_textbox = tk.Text(self.input_textbox_frame, height=20, width=text_width)
        textbox_scrollbar = tk.Scrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox["yscrollcommand"] = textbox_scrollbar.set
        textbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.input_textbox.pack()

        self.add_to_queue_button = tk.Button(self, text="Add to Queue", command=self.add_to_queue)
        self.add_to_queue_button.pack()

        self.queue_label = tk.Label(self, text="Queue length: 0")
        self.queue_label.pack()

        self.audio_queue_label = tk.Label(self, text="Audio queue length: 0")
        self.audio_queue_label.pack()

        self.message_textbox_frame = tk.Frame(self)
        self.message_textbox_frame.pack()
        self.message_textbox_label = tk.Label(self.message_textbox_frame, text="Messages:")
        self.message_textbox = tk.Text(self.message_textbox_frame, height=32, width=text_width, state=tk.DISABLED)
        message_textbox_scrollbar = tk.Scrollbar(self.message_textbox_frame, command=self.message_textbox.yview)
        self.message_textbox["yscrollcommand"] = message_textbox_scrollbar.set
        message_textbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.message_textbox.pack()

        # Start the worker threads.
        self.text_to_audio_worker = threading.Thread(target=text_to_audio_worker, args=(self.text_queue, self.audio_queue, self.worker_message_queue))
        self.text_to_audio_worker.start()
        self.audio_speaking_worker = threading.Thread(target=audio_speaking_worker, args=(self.audio_queue, self.worker_message_queue))
        self.audio_speaking_worker.start()

        # Ensure the queue_label gets updated regularly.
        self.label_update_delay = 250  # ms
        self.after(self.label_update_delay, self.update_queue_label)
        self.after(self.label_update_delay, self.update_audio_queue_label)
        self.after(self.label_update_delay, self.update_message_label)

    def update_queue_label(self):
        self.queue_label["text"] = f"TTS queue length: {self.text_queue.qsize()}"
        self.after(self.label_update_delay, self.update_queue_label)

    def update_audio_queue_label(self):
        self.audio_queue_label["text"] = f"Speaker queue length: {self.audio_queue.qsize()}"
        self.after(self.label_update_delay, self.update_audio_queue_label)

    def update_message_label(self):
        if not self.worker_message_queue.empty():
            message = self.worker_message_queue.get()
            # Append the message (after newline) to our message box existing text, and scroll to the bottom.
            self.message_textbox.configure(state=tk.NORMAL)
            self.message_textbox.insert(tk.END, message + "\n")
            self.message_textbox.see(tk.END)
            self.message_textbox.configure(state=tk.DISABLED)
            
        self.after(self.label_update_delay, self.update_message_label)
        
    def add_to_queue(self):
        text = self.input_textbox.get("1.0", "end-1c")
        for chunk in breakup(text):
            self.text_queue.put(chunk)
        self.queue_label["text"] = f"Queue length: {self.text_queue.qsize()}"

        # Clear the input box.
        self.input_textbox.delete("1.0", tk.END)

    def destroy(self):
        # Clicking the close button should just kill the whole process, threads be damned.
        super().destroy()
        import os
        os._exit(0)
        

if __name__ == "__main__":
    print("Starting app...")
    app = SpeakerApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Exiting app due to KeyboardInterrupt.")
        app.destroy()
