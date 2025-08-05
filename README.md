---
title: Rave Model Averager
license: cc-by-nc-4.0
emoji: 📻
colorFrom: purple
colorTo: green
sdk: gradio
python_version: 3.12
sdk_version: 5.39.0
app_file: app.py
suggested_hardware: cpu-basic
pinned: false
short_description: Encodes/decodes audio through the average of two RAVE models
models: 
    - Intelligent-Instruments-Lab/rave-models
    - shuoyang-zheng/jaspers-rave-models
preload_from_hub:
    - Intelligent-Instruments-Lab/rave-models 
    - shuoyang-zheng/jaspers-rave-models]
tags: 
    - RAVE
    - Audio 
    - Model Manipulation
    - Encode
    - Decode
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Built using:
Mac OS Sequoia 15.5
Python 3.12

This app attempts to average two RAVE models and then encode and decode an audio file through the original and averaged models.

Instructions:
- Select the two models from the lists of pre-trained RAVE models.
- Select an audio file from the ones available in the dropdown, or upload an audio file of up to 60 seconds. Click 'Submit' at the bottom of the page.

Notes:
- Generally, the audio encoded/decoded using the average model does not sound like an 'average' of the two models. One of the better examples comes from the default settings, where you can here the sounds from model A (multi-timbral guitar) modulated somewhat by model B (water). 
- The versions encoded/decoded through the individual models give interesting results though.
- In most cases not all parameters can be averaged. They may not exist in both models, or they may not have the same shape. The data sets in the output list which ones were and weren't averaged with their shapes and any notes. (You can copy them into a spreadsheet by clicking the icon in the top right corner of each.)
- The averaged model starts as a clone of Model A. Parameters that can't be averaged default to Model A values.
- If all the parameters can be averaged, the result is usually not good - a high pitch squeal or a low rumble. 


Outstanding questions for interested parties:
- Since it doesn't work well when all params are compatible, mayber there are some params that shouldn't be averaged?
- Would it make logical sense to reshape the parameters that exist in both models but do not have the same shape so they can be averaged? 
- Anything thoughts on what could make the results sonically more like an 'average' of two models?

