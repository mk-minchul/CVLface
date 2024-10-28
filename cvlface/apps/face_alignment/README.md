# Face Alignment App

## Overview
The Face Alignment App is designed to process facial images by aligning them to a canonical position suitable for face recognition purposes. The app aligns each face and resizes the image to a standard size of 112x112 pixels.

## Directory Structure
Ensure your project directory is structured as follows:

```
├── images/                 # Directory containing original unaligned images
│   ├── image1.jpg
│   ├── subfolder/image2.png
│   ...
│
└── aligned_images/         # Directory where the aligned images will be saved
│   ├── image1.jpg
│   ├── subfolder/image2.png
```

Run the script using the following command:
```bash
python align_faces.py --aligner_id minchul/cvlface_DFA_mobilenet --data_root ./example/images --save_root ./example/aligned_images
```

### Arguments
- `--aligner_id`: Identifier for the alignment model. Default is 'minchul/cvlface_DFA_mobilenet'.
- `--data_root`: Path to the directory containing the images to be aligned. Default is './example/images'.
- `--save_root`: Path where the aligned images will be saved. Default is './example/aligned_images'.

### Expected Results
After running the script, aligned images will be saved in the `aligned_images` directory. Each image will be resized to 112x112 pixels to meet the requirements for face recognition.


<table align="center">
<tr>
<td><img src="example/images/3.png" alt="Image 1"></td>
<td><img src="example/aligned_images/3.png" alt="Image 1"></td>
<td><img src="example/aligned_images/3_ldmks.png" alt="Image 2"></td>
</tr>
<tr>
<td><img src="example/images/2.png" alt="Image 1"></td>
<td><img src="example/aligned_images/2.png" alt="Image 1"></td>
<td><img src="example/aligned_images/2_ldmks.png" alt="Image 2"></td>
</tr>
</table>