import huggingface_hub
# 
# paths to various models
model_path_configs = {
        "Humpback Whales":      ("Intelligent-Instruments-Lab/rave-models", "humpbacks_pondbrain_b2048_r48000_z20.ts"), 
        "Magnets":              ("Intelligent-Instruments-Lab/rave-models", "magnets_b2048_r48000_z8.ts"), 
        "Big Ensemble":         ("Intelligent-Instruments-Lab/rave-models", "crozzoli_bigensemblesmusic_18d.ts"),
        "Bird Dawn Chorus":     ("Intelligent-Instruments-Lab/rave-models", "birds_dawnchorus_b2048_r48000_z8.ts"), 
        "Speaking & Singing":   ("Intelligent-Instruments-Lab/rave-models", "voice-multi-b2048-r48000-z11.ts"), 
        "Resonator Piano":      ("Intelligent-Instruments-Lab/rave-models", "mrp_strengjavera_b2048_r44100_z16.ts"),
        "Multimbral Guitar":    ("Intelligent-Instruments-Lab/rave-models", "guitar_iil_b2048_r48000_z16.ts"),
        "Organ Archive":        ("Intelligent-Instruments-Lab/rave-models", "organ_archive_b2048_r48000_z16.ts"),
        "Water":                ("Intelligent-Instruments-Lab/rave-models", "water_pondbrain_b2048_r48000_z16.ts"),
        "Brass Sax":            ("shuoyang-zheng/jaspers-rave-models", "aam_brass_sax_b2048_r44100_z8_noncausal.ts"),
        "Speech":               ("shuoyang-zheng/jaspers-rave-models", "librispeech100_b2048_r44100_z8_noncausal.ts"),
        "String":               ("shuoyang-zheng/jaspers-rave-models" ,"aam_string_b2048_r44100_z16_noncausal.ts"),
        "Singer":               ("shuoyang-zheng/jaspers-rave-models","gtsinger_b2048_r44100_z16_noncausal.ts"),
        "Bass":                 ("shuoyang-zheng/jaspers-rave-models","aam_bass_b2048_r44100_z16_noncausal.ts"),
        "Drum":                 ("shuoyang-zheng/jaspers-rave-models","aam_drum_b2048_r44100_z16_noncausal.ts"),
        "Gtr Picking":          ("shuoyang-zheng/jaspers-rave-models","guitar_picking_dm_b2048_r44100_z8_causal.ts"),
    }

available_audio_files=[
    "SilverCaneAbbey-Voices.wav",
    "Chimes.wav",
    "FrenchChildren.wav",
    "Organ-ND.wav",
    "SpigotsOfChateauLEtoge.wav",
    "GesturesPercStrings.wav",
    "SingingBowl-OmniMic.wav",
    "BirdCalls.mp3",
    ]

model_path_config_keys = sorted(model_path_configs)
model_paths_cache = {}

def GetModelPath(model_path_name):
    model_path = ()

    if model_path_name in model_paths_cache.keys():
        model_path = model_paths_cache[model_path_name]
    else:
        repo_id, filename = model_path_configs[model_path_name]
        
        model_path = huggingface_hub.hf_hub_download(
        repo_id =repo_id,
        filename = filename,
        cache_dir="../huggingface_hub_cache",
        force_download=False,
        )
        
        print(f"Generated Model Path for {filename}.")
        model_paths_cache[model_path_name] = model_path
        
    return model_path 

def saveAudio(file_path, audio):
    with open(file_path + '.wav', 'wb') as f:
        f.write(audio.data)
        
import torch
import pandas as pd
import copy
import librosa
import ast
import os

def AverageRaveModels(rave_a, rave_b, bias = 0):

    r1_ratio = .5
    r2_ratio = .5

    messages = {}
    # bias between -1 and 1
    if abs(bias) <= 1:
        if bias > 0:
            r1_ratio = .5 + bias/2
            r2_ratio = 1.0 - r1_ratio

            rave_temp = rave_a
        elif bias < 0:
            r2_ratio = .5 + abs(bias)/2
            r1_ratio = 1.0 - r2_ratio
    else:
        print(f"Unable to apply bias {bias} - bias must be between -1 and 1.")
    
    # Get state dictionaries of both models
    rave_a_params = rave_a.state_dict()
    rave_b_params = rave_b.state_dict()
    
    # intialize the averaged rave with model_a
    rave_avg = copy.deepcopy(rave_a)
    avg = rave_avg.state_dict()    

    # for reporting
    keys_averaged={}
    keys_not_averaged={}
    for key in rave_a_params:
        if key in rave_b_params:
            try:
                avg[key] = ((rave_a_params[key] * r1_ratio) + (rave_b_params[key] * r2_ratio)) 
                keys_averaged[key]=(key, rave_a_params[key].shape, rave_b_params[key].shape, "")
            except Exception as e:
                print(f"Error averaging key {key}: {e}")
                keys_not_averaged[key]=(key, rave_a_params[key].shape, rave_b_params[key].shape, e)
        else:
            print(f"Key {key} not found in rave_b parameters, skipping.")
            # keys_not_averaged(key)
            keys_not_averaged[key]=(key, rave_a_params[key].shape, "n/a", "Key not found in rave_b parameters.")
        
    messages["keys_averaged"] = keys_averaged
    messages["keys_not_averaged"] = keys_not_averaged

    messages["stats"] = f'Numb Params Averaged: {len(keys_averaged)}\nNumb Params Unable to Average: {len(keys_not_averaged)}\nPercent Averaged: {len(keys_averaged) * 100/(len(keys_not_averaged) + len(keys_averaged)):5.2f}%'
    
    # Commit the changes
    rave_avg.load_state_dict(avg) 
    
    return rave_avg, messages

def GenerateRaveEncDecAudio(model_name_a, model_name_b, audio_file_name, audio_file, sr_multiple=1, bias=0): #audio_file_name="RJM1240-Gestures.wav"

    ###############################################
    # Choose models from filenames dictionary created in previous cell
    # Note: model_path_a is always used to initialize the averaged model.
    # Switching them gets different results if the parameters are not all matched.
    ###############################################
    # Examples - this matches only 21 params, but it sounds like maybe sosme of both are in the result.
    model_path_a = GetModelPath(model_name_a)
    model_path_b = GetModelPath(model_name_b)

    # Examples: This has 76 params averaged
    # model_path_a = model_paths['Water']
    # model_path_b = model_paths['Organ Archive']

    # Examples: All Params Match but high pitch for averaged version
    # model_path_a = model_paths['Organ Archive']
    # model_path_b = model_paths['Multimbral Guitar']
    #
    # model_path_a = model_paths['String']
    # model_path_b = model_paths['Singer']
    #
    # Examples - All Params Match but get a lower frequency effect
    # model_path_a = model_paths['Whale']
    # model_path_b = model_paths['Water']


    #####################################
    # Set biases between -1 and 1 to bias the result towards one of the models
    #   0 = no bias; >0  biased towards model_a; <0 = biased towards  model_b
    #####################################
    # Note: multiple biases not implemented for gradio version
    biases=[bias]

    ####################################
    # Choose Audio File to encode/decode
    #####################################
    # audio_file_name = "RJM1240-Gestures.wav"
    if audio_file is None:
        audio_file = os.path.join('assets', audio_file_name)
    # print("Audio File Name:", audio_file_name)


    ####################################
    # Generate Audio Files
    # Audio files are created in the assets folder
    generate_audio_files = False

    rave_a = torch.jit.load(model_path_a)
    rave_b = torch.jit.load(model_path_b)

    # Let's load a sample audio file
    y, sr = librosa.load(audio_file)

    sr_multiplied = sr * sr_multiple  # Adjust sample rate if needed
    print(f"Audio File Loaded: {audio_file}, sample_rate = {sr}")
   
    # Convert audio to a PyTorch tensor and reshape it to the
    # required shape: (batch_size, n_channels, n_samples)
    audio = torch.from_numpy(y).float()
    audio = audio.reshape(1, 1, -1) 

    messages={}
    audio_outputs={}
    for bias in biases:
        # Average the rave models
        # rave_avg, numb_params_mod, numb_params_unable_to_mod = AverageRaveModels(rave_a, rave_b, bias=bias)
        rave_avg, new_msgs = AverageRaveModels(rave_a, rave_b, (-1 * bias))
        messages |= new_msgs 

        # no decode the results back to audio
        with torch.no_grad():
            # encode the audio with the new averaged models
            try:
                latent_a = rave_a.encode(audio)
                latent_b = rave_b.encode(audio)
                latent_avg = rave_avg.encode(audio)

                # decode individual and averaged models
                decoded_a = rave_a.decode(latent_a)
                decoded_b = rave_b.decode(latent_b)
                decoded_avg = rave_avg.decode(latent_avg)
                audio_outputs[bias] = decoded_avg[0]
            except:
                print(f'Bias {bias} generated an error. Removing it from list of biases.')
                biases.remove(bias)
                # print(biases)
                
        model_a_file=model_path_a.rsplit("/")[-1]
        model_b_file=model_path_b.rsplit("/")[-1]

        # Original Audio
        original_audio = (sr, y)

        # Decoded Audio
        print("Encoded and Decoded using original models")
        model_a_audio =  (sr, decoded_a[0].detach().numpy().squeeze())
        # saveAudio('assets/' + model_a_file[: 7] + '_only.wav', a)

        model_b_audio = (sr, decoded_b[0].detach().numpy().squeeze())
        # # saveAudio('assets/' + model_b_file[: 7] + '_only.wav', a)

        print("Encoded and Decoded using Averaged Models")
        print("with Biases: ", biases)
        print("\nNumber of params able to average:", len(messages["keys_averaged"]))
        print("Number of params unable to average:", len(messages["keys_not_averaged"]))

        output_file_prefix = f'assets/{model_a_file[: 7]}-{model_b_file[: 7]}_'
        
        bias = biases[0]
        averaged_audio = (sr_multiplied, audio_outputs[bias].detach().numpy().squeeze()) 
        
        df_averaged = pd.DataFrame(messages['keys_averaged']).transpose() #reset_index(names='Param Key')
        df_averaged.columns=['Param Name', 'Model A Shape', 'Model B Shape', 'Notes']
        
        df_not_averaged = pd.DataFrame(messages["keys_not_averaged"]).transpose()
        
        # case when all params are averaged
        if len(df_not_averaged.columns) == 0:
            data = {'Param Name': [], 'Modeal A Shape': [], 'Model B Shape': [], 'Notes': []}
            df_not_averaged = pd.DataFrame(data)
    
        df_not_averaged.columns=['Param Name', 'Model A Shape', 'Model B Shape', 'Notes']

        messages["stats"] = f"Model A: {model_name_a}\nModel B: {model_name_b}\nAudio file: {os.path.basename(audio_file)}\nSample Rate Multiple for Averaged Version: {sr_multiple}\n\n" + messages["stats"]
        
        return original_audio, model_a_audio, model_b_audio, averaged_audio, messages["stats"], df_averaged, df_not_averaged
        
import gradio as gr

column_widths=['35%', '20%', '20%', '25%']
waveform_options = gr.WaveformOptions(waveform_color="#01C6FF", 
                                                     waveform_progress_color="#0066B4",
                                                     skip_length=2,)

description = "<p style='line-height: 1'>This app attempts to average two RAVE models and then encode and decode an audio file through the original and averaged models.</p>" \
"<ul style='padding-bottom: 0px'>Instructions:<li style='line-height: 1; padding-top: 5px'>Select the two models from the lists of pre-trained RAVE models.</li>" \
"<li style='line-height: 1; padding-top: 0px'>Select an audio file from the ones available in the dropdown, or upload an audio file of up to 60 seconds. Click 'Submit' at the bottom of the page.</li></ul>"\
"<p style='line-height: 1.2; padding-top: 0px; margin-top: 3px;'>Note that in most cases not all parameters can be averaged. They may not exist in both models or the two values may not have the same shape. The data sets in the output list which ones were and weren't averaged with their shapes and any notes. (You can copy them into a spreadsheet by clicking the icon at the top right corner of each.)</p>"

"<!-- <li>Select a sample rate multiple for the averaged model. When there is a useful result, it sometimes sounds better at double the sample rate.</li>" \
"<li>Select a bias towards one of the models. A bias of 0 will average the two models equally. A positive bias will favor Model A, and a negative bias will favor Model B.</li></ul>" \
"-->"



AverageModels = gr.Interface(title="Process Audio Through the Average of Two Rave Models", description=description,
    fn=GenerateRaveEncDecAudio,
    inputs=[
        gr.Radio(model_path_config_keys, label="Select Model A", value="Multimbral Guitar", container=True),
        gr.Radio(model_path_config_keys, label="Select Model B", value="Water", container=True),
        gr.Dropdown(available_audio_files, label="Select from these audio files or upload your own below:", value="SilverCaneAbbey-Voices.wav",container=True),
        gr.Audio(label="Upload an audio file (wav)", type="filepath", sources=["upload", "microphone"], max_length=60,
                waveform_options=waveform_options, format='wav'),],
    additional_inputs=[
        gr.Radio([.2, .5, .75, 1, 2, 4], label="Sample Rate Multiple (Averaged version only)", value=1, container=True),
        gr.Slider(label="Bias towards Model A or B", minimum=-1, maximum=1, value=0, step=0.1, container=True),
        ],
    # if no way to pass dictionary, pass separate keys and values and zip them.
    outputs=[
        gr.Audio(label="Original Audio", sources=None, waveform_options=waveform_options, interactive=False),
        gr.Audio(label="Encoded/Decoded through Model A", sources=None, waveform_options=waveform_options,),
        gr.Audio(label="Encoded/Decoded through Model B", sources=None, waveform_options=waveform_options,),
        gr.Audio(label="Encoded/Decoded through averaged model", sources=None, waveform_options=waveform_options,),
        gr.Textbox(label="Info:"),
        gr.Dataframe(label="Params Averaged", show_copy_button="True", scale=100, column_widths=column_widths, headers=['Param Name', 'Model A Shape', 'Model B Shape', 'Notes']),
        gr.Dataframe(label="Params Not Averaged", show_copy_button="True", scale=100, column_widths=column_widths, headers=['Param Name', 'Model A Shape', 'Model B Shape', 'Notes'])
        ]
    ,fill_width=True
)

AverageModels.launch(max_file_size=10 * gr.FileSize.MB, share=True)