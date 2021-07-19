# Eyeglance Detection

This is a fine-tuned model based on ResNet18 and trained on the SHRP2 Baseline and Crash dataset to estimate head pose by category. It can identify six categories "Center stack, Cup holder - console", "Forward, Instrument cluster, Left windshield", "Left window / mirror", "Rearview mirror", "Right window / mirror", "Right windshield".

## Setup
The dependencies are

- python3
- pytorch
- torchvision
- Pillow

I suggest using a python virtualenv to run the code.

    virtualenv vpy
    . vpy/bin/activate
    pip install -r requirements.txt

## Running the model

Download the model with curl 

    curl -JLO https://mirror.vtti.vt.edu/vtti/ctbs/eyeglance/ResNet18_batchsize-96_trainlayers-all_fctype-linear_int_fcint-300_learningrate-0.0001_lrdecay-0.1_lrdecaystep-7_epochs-15_optimizer-sgd_model-resnet18_cropping-true_balanced-true_rebalanced-true.pth

You can run the model on a cropped face image using the `run_image.py` command.

    python run_image.py ResNet18_batchsize-96_trainlayers-all_fctype-linear_int_fcint-300_learningrate-0.0001_lrdecay-0.1_lrdecaystep-7_epochs-15_optimizer-sgd_model-resnet18_cropping-true_balanced-true_rebalanced-true.pth examples/3792_06-25_Console.png

Retinaface was used for cropping faces from images.

There are a few example images in the `examples` directory taken from the [SHRP2 Safety Pilot Study](https://insight.shrp2nds.us/)
