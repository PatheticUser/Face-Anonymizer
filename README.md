
# Face-Anonymizer 

A lightweight Python tool to automatically detect and anonymize faces in images and videos. Perfect for privacy-conscious developers, journalists, and content creators!  

---

## Features

- Automatically detects faces in images and videos using OpenCV or deep learning models.
- Anonymizes faces with blur or pixelation.
- Supports batch processing of multiple images.
- Easy to integrate into Python scripts or pipelines.
- Minimal setup and dependencies.

---

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/PatheticUser/Face-Anonymizer.git
cd Face-Anonymizer
pip install -r requirements.txt
````

> Make sure you have Python 3.7 or higher installed.

---

## Usage

### Anonymize an image

```bash
python anonymize.py --input path/to/image.jpg --output path/to/output.jpg --method blur
```

### Anonymize a video

```bash
python anonymize.py --input path/to/video.mp4 --output path/to/output.mp4 --method pixelate
```

### Options

* `--input` : Path to the input image or video.
* `--output` : Path to save the anonymized output.
* `--method` : Anonymization method (`blur` or `pixelate`).

---

## How It Works

1. The program uses OpenCV’s face detection (Haar cascades or DNN) to locate faces.
2. Each detected face is processed with the chosen anonymization method:

   * **Blur:** Applies a Gaussian blur over the face region.
   * **Pixelate:** Reduces face resolution for a mosaic effect.
3. The output is saved in the specified folder.

---

## Use Cases

* Protecting privacy in public datasets.
* Blurring faces in videos for social media content.
* Assisting journalists and researchers with anonymized media.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/PatheticUser/Face-Anonymizer/issues).

1. Fork the repo
2. Create a new branch (`git checkout -b feature-name`)
3. Make your changes
4. Commit (`git commit -m 'Add some feature'`)
5. Push (`git push origin feature-name`)
6. Open a pull request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

* [OpenCV](https://opencv.org/) for face detection and image processing.
* Inspired by the need for privacy-first media handling tools.

```