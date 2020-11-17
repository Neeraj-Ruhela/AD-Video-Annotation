Video attribution is a multi-label classification problem that aims at converting the video files into formats like audio, text, and image and extract the semantics of the detected object.
![image](https://user-images.githubusercontent.com/71966691/99437588-b1090d80-2938-11eb-868a-a67307b4ed37.png)

Spending on digital advertising has risen each year even as TV and print advertising declines.
Understanding the value of digital advertising requires understanding of how value is
attributed to each activity. Understanding the true worth of these channels is key to any
company’s success.
What an advertiser wants to know is how digital advertising really affects the bottom line.
You spend more, do you get more? How do you know what works and what does not?
Do combinations of advertising work best? How does advertising work?
Now a days there are ample promotional videos available in the market to analyze so before launching a new product client has to analyze history of events and understand customer acceptance. By analyzing these videos and its sentiments client may strategize market penetration and get benefit out of it. To do so there are lots of man force required who will watch the video and create notes for the events in it.
However due to lots of data it is a tedious job and involves cost. And also due to different skill set it is challenging. Goal is to automate the whole process of video annotation. 

I have divided the projects in 4 tasks.

## TASK 1: - Frame and Audio Extraction
•	Video frame extraction, Audio extraction, Grey scaling extracted video frames.

### 1.1 Problem Statement:
To extract the text from video and audio available in video file we need to get the frames and audio file separately. In this chapter we are doing the pre-processing steps for text extraction.

### 1.2 Objective
Once we extract video frames and audio from ad-video, further we will get the text from both the sources and perform text analysis. But before extracting text from video frames we need to convert them to grey scale image.
. The key benefit of these steps are:
1.	This will input in further steps like optical character recognition (OCR) from images.
2.	Also audio file will be used in text extraction and further analyzing headlines and taglines

### 1.3 Literature Review
Data mining is a process of extracting previously unknown knowledge and detecting the interesting patterns from a massive set of data. Thanks to the extensive use of information technology and the recent developments in multimedia systems, the amount of multimedia data available to users has increased exponentially. Video is an example of multimedia data as it contains several kinds of data such as text, image, meta-data, visual and audio. It is widely used in many major potential applications like security and surveillance, entertainment, medicine, education programs and sports. The objective of video data mining is to discover and describe interesting patterns from the huge amount of video data.

### 1.4 Machine Learning Process Flow:
![image](https://user-images.githubusercontent.com/71966691/99430658-a72edc80-292f-11eb-9f43-277854713066.png)

### 1.5 Resources
Software: - OpenCV, Pydub, moviepy

### 1.6 Potential Data Challenges and Risks
1.	Creating proper folder structuring for initial video, video frames and audio file
2.	Specifying the time frame for video frames extraction so that we should not miss any important information.
3.	Defining the video file format that to be accepted.
4.	Deciding the shape of frames.
5.	Deciding the frame rate of audio file.

### 1.7 Detailed Plan of Work
•	Extract frames from the video advertisements.
•	Convert the frames to grey scale images.
•	Extract audio from ad-video. 

### 1.8 Pre-Processing Steps
•	Proper folder structuring for root folder, initial video, video frames and audio file.
•	If the folders are not present, then accordingly code to create so.
•	Defining file format in python code for acceptance.
•	Converting the extracted frames to gray scale.
•	Decided the shape of extracted frames.

### 1.9 Machine Learning Modelling and Techniques Applied
Moviepy: this python library is used for video editing, cutting, concatenations, title insertions and creation of custom effects. 
Opencv: It is basically used for computer vision applications. It mainly focuses on image processing, video capture and analysis including features like object & face detection.
Pydub: This library is used for audio manipulation.

### 1.10 Important Points
•	Frame rate has been set to 16Khz.
•	Frame
extraction is 1frame/sec; this can be changed as per use case.
•	Fixed the shape of extracted gray frames to 
  Frame width=2000, Frame height=2000
  
### Task1_Result: - 
![image](https://user-images.githubusercontent.com/71966691/99429629-44891100-292e-11eb-9e1f-994248c1f065.png)


## TASK 2: -Optical Character Recognition

### 2.1 Problem Statement
We all need to analyze and search for text at some point or other. We can have various sources in which we would want to search or analyze text like documents (pdf or word) or images. Here we will be extracting text from frames in order to get the headlines and tag line of the video.

### 2.2 Objective
With Optical Character Recognition(OCR), our objective is to detect, search and analyze text present in documents or in images. For our projects, we are doing OCR on images extracted as frames from the video advertisements. The key benefit we get using OCR are:
•	Analysis of text with detailed insights
•	Helps in doing market analysis (in case of advertisements) to help organization decide popular products
•	Increases efficiency and quality of text analysis.
•	Better visualization of text.

### 2.3 Literature Review
The roots of OCR can be traced back to systems before the inventions of computers. The earliest OCR systems were mechanical devices that were able to recognize characters or text, but had very slow speeds and also low accuracies. In 1951, M. Sheppard invented a reading robot named GISMO that could be considered as the earliest work on modern OCR. GISMO could read musical notations as well as words on a printed page one by one. However, it could only recognize only a few characters, 23 to be precise. The machine also had the capability to copy a typewritten page. In 1954, J. Rainbow devised a machine that could read uppercase typewritten English characters, one per minute. However, the early OCR systems were criticized due to errors and slow recognition speeds. Hence, not much research efforts were put on the topic during the 60's and the 70’s. The only developments were done on government agencies and large corporations like banks, newspapers and airlines etc. Because of the complexities associated with recognition, it was felt that there should be standardized OCR fonts for easing the task of recognition for OCR. Hence, OCRA and OCRB were developed by ANSI and
EMCA in 1970, that provided comparatively acceptable recognition rates. During the past thirty years, substantial research has been done on OCR. This has led to the emergence of document image analysis (DIA), multi-lingual, handwritten and omni-font OCRs. 

### 2.4 Machine Learning Process Flow
![image](https://user-images.githubusercontent.com/71966691/99430206-0e985c80-292f-11eb-8b2a-18ecbac3f90c.png)

Detection part is using the CRAFT (Character Region Awareness for Text Detection) algorithm. Its pre-trained model has also been used.

Recognition model is CRNN. It is composed of 3 main components:
1.	Feature extraction (using Resnet)
2.	Sequence labeling (LSTM).
3.	Decoding (CTC). Word beam search decoding is a Connectionist Temporal Classification (CTC) decoding algorithm. It is used for sequence recognition tasks like Handwritten Text Recognition (HTR) or Automatic Speech Recognition (ASR). The following illustration shows a HTR system with its Convolutional Neural Network (CNN) layers, Recurrent Neural Network (RNN) layers and the final CTC (loss and decoding) layer. Word beam search decoding is placed right after the RNN layers to decode the output, see the red dashed rectangle in the illustration.
### Handwritten Text Recognition (HTR) with CNN, RNN and CTC.
![image](https://user-images.githubusercontent.com/71966691/99430889-efe69580-292f-11eb-99f5-ffd626f0d3e2.png)
Training pipeline for the recognition part is a modified version from deep-text-recognition-benchmark. Data synthesis is based on TextRecognitionDataGenerator.

### 2.5 Resources
Software- Easy OCR, Google Colab, xlsxwriter

### 2.6 Potential Data Challenges and Risks
•	Quality of documents and images is very important. Poor quality leads to very poor results.
•	Does not work well with colored images. That is why frames extracted from video advertisements used in this project have been converted to greyscale as the pixel difference between background and text in a gray-scaled image is significant for the model to detect text.
•	Does not work well with handwritten text.
•	Does not work well with images affected by artifacts including partial occlusion, distorted perspective, and complex background.

### 2.7 Detailed Plan of Work
•	Extract frames from the video advertisements.
•	Convert the frames to grey scale images.
•	Detect text from the grey scale frames using Easy OCR. 
•	Prepare a worksheet (excel) depicting the extracted text against the frame number, separated by the video name of the corresponding frames.
•	Compare the extracted text from the worksheet with the one present in the corresponding frame, and check the accuracy.

### 2.8 Pre-Processing Steps
•	Extract frames from all the video advertisements being used.
•	Organize the frames according to their corresponding video advertisements.
•	Convert these frames into grey scale, video-wise.
•	Easy OCR will apply the CRAFT algorithm to detect text. CRAFT is a scene text detection method to effectively detect text area by exploring each character and affinity between the characters.

### 2.9 Machine Learning Modelling and Techniques Applied
Easy OCR uses the following machine learning algorithms in various stages for Optical Character Recognition:
1.	It uses the CRAFT algorithm for text detection. CRAFT is a scene text detection method to effectively detect text area by exploring each character and affinity between the characters.
2.	Text recognition model used is CRNN (Convolutional Recurrent Neural Network). It further has three main components:
a.	Feature Extraction: Feature extraction for the recognized text, like font of the text, style, size, spacing between characters of a text, etc. using Google ResNet.
b.	Sequence Labelling: Sequence Labelling is done to keep the sentence intact when extracting the text. For this, Long Short Term Memory (LSTM) algorithm is used.
c.	Decoding: Decoding is used for sequence recognition task, like Handwritten Text Recognition (HTR) or Automatic Speech Recognition (ASR). For this, Easy OCR uses the Connectionist Temporal Classification (CTC) decoding algorithm.

### 1.10 Important Points
•	Probability of extracted text for each bounding box is used to check the confidence level of extraction.

### Task2_Result:
![image](https://user-images.githubusercontent.com/71966691/99431481-d8f47300-2930-11eb-812f-bfcd3abe6180.png)


## Task3: - Speech to text conversion and Finding Head line and Tag line
### 3.1 Problem statement: - 
To extract the information in textual format from the audio content available in video.

### 3.2 Objective of the project: - 
To extract the text with highest possible accuracy level. This text will be used for finding the Headlines and Taglines. This process is also often called speech recognition. Although these terms are almost synonymous, Speech recognition is sometimes used to describe the wider process of extracting meaning from speech, i.e. speech understanding. The term voice recognition should be avoided as it is often associated to the process of identifying a person from their voice, i.e. speaker recognition.

### 3.3 Background of previous work done in the chosen area: -
 The speech to text problem is complex in nature. The speech comes with natural variations from person to person and background noise also affects the quality of speech content. Two different audios (with different fame rate, background noise, pitch and tone etc.) might have same content, this proves the complexity of problem.
There has been a significant improvement in speech to text conversion, current voice assistants like Amazon Alexa and Google Home are becoming very popular these days, they have changed the way we interact with our electronic device, how we shop online, how we do web search.

### 3.4 Machine Leaning process flow: - 
It comprises of two models an acoustic model and a language model. Large vocabulary systems also use a pronunciation model. No single system can transcribe all the languages i.e. the Speech to text models (speech recognizer) are language specific. The transcription quality also depends on the speaker, the style of speech and the environmental conditions, dialect, application domain, type of speech, and communication channel. Speech to text conversion is useful in many applications like controlling devices using spoken words, audio document transcription etc. Each use has specific requirements in terms of latency, memory constraints, vocabulary size, and adaptive features.
![image](https://user-images.githubusercontent.com/71966691/99431627-122ce300-2931-11eb-96b8-51d6d0c70ee9.png)

### Language model: 
A language model captures the regularities in the spoken language and is used by the speech recognizer to estimate the probability of word sequences. One of the most popular method is the so called n-gram model, which attempts to capture the syntactic and semantic constraints of the language by estimating the frequencies of sequences of n words.

### Acoustic model: 
A model describing the probabilistic behavior of the encoding of the linguistic information in a speech signal. LVCSR systems use acoustic units corresponding to phones or phones in context. The most predominant approach uses continuous density hidden Markov models (HMM) to represent context dependent phones.

LVCSR: Large Vocabulary Speech Recognition (large vocabulary means 20k words or more). The size of the recognition vocabulary affects the processing requirements.

### 3.5 Resources needed for the project: -
Software: - google colab and IBM Watson for speech to text conversion.

### 3.6 Potential data challenges & risks in doing the project: -
Audio data quality depends on the speaker, the style of speech and the environmental conditions, dialect, application domain, type of speech, and communication channel.
The speech to text conversion problem requires a huge amount of labeled data for training of models. As we don’t have that much amount of data, we are using the pre-trained model. The option for customization of language and acoustic model as per the use case is also available but that requires huge amount of data.

### 3.7 Detailed Plan of Work: - 
We will be using various libraries (Open source and Trail Version) available for speech to text conversion.
We will fine tune the all the available models and will compare the result. Model which will give the best result for our use case we will be adding that in our final code.

The various options that we tried are discussed below:

#### 1. DeepSpeech (pip3 install deepspeech) : -
 DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on Baidu’s Deep Speech research paper. 
 It is an open source library. Also has good detailed documentation to custom train your models. 
 The transcription quality was not good for initial sentences in our use case. This model is sensitive to framerate of audio and doesn’t provide auto adjust to frame rate. 
 The complete RNN model is illustrated in the figure below.
 
 ![image](https://user-images.githubusercontent.com/71966691/99431834-5cae5f80-2931-11eb-9a70-ee0d4dc369b5.png)
 
#### 2. SpeechRecognition (pip install SpeechRecognition): -
Library for performing speech recognition. Architecture of this library is not open source. Not many hyper parameters to tune the pre-trained model. 
The quality of transcription was not very good for end sentences in our use case.

#### 3. IBM Watson(pip install ibm-watson): - (Preferred)
The Speech to Text service provides an API to add speech transcription capabilities to applications.
It combines information about language structure with the composition of the audio signal.

#### Supported language models:
•	Broadband models are for audio that is sampled at greater than or equal to 16 kHz. Use broadband models for responsive, real-time applications, for example, for live-speech applications.
•	Narrowband models are for audio that is sampled at 8 kHz. Use narrowband models for offline decoding of telephone speech, which is the typical use for this sampling rate.
![image](https://user-images.githubusercontent.com/71966691/99432563-553b8600-2932-11eb-8622-3c23a53360fc.png)
Link for complete models list- https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-models

#### Language and model for our application
                            Language: - English (United States)         &         Narrow band Model: - en-US_ShortForm_NarrowbandModel   

The US English short-form model, en-US_ShortForm_NarrowbandModel, can improve speech recognition for Interactive Voice Response (IVR) and Automated Customer Support solutions. The short-form model is trained to recognize the short utterances that are frequently expressed in customer support settings like automated support call centers. In addition to being tuned for short utterances in general, the model is also tuned for precise utterances such as digits, single-character word and name spellings, and yes-no responses.
The en-US_ShortForm_NarrowbandModel is optimal for the kinds of responses that are common to human-to-machine exchanges, such as the use case of IBM® Voice Agent with Watson™. The en-US_NarrowbandModel is generally optimal for human-to-human conversations. However, depending on the use case and the nature of the exchange, some users might find the short-form model suitable for human-to-human conversations as well. Given this flexibility and overlap, you might experiment with both models to determine which works best for your application. In either case, applying a custom language model with a grammar to the short-form model can further improve recognition results.
As with all models, noisy environments can adversely impact the results. For example, background acoustic noise from airports, moving vehicles, conference rooms, and multiple speakers can reduce transcription accuracy. Audio from speaker phones can also reduce accuracy due to the echo common to such devices. Using the parameters available for speech activity detection can counteract such effects and help improve speech transcription accuracy. Applying a custom acoustic model can further fine-tune the acoustics for speech recognition, but only as a final measure.

### 3.8 Pre-Processing Steps (Input features)
Sampling rate: - Sampling rate (or sampling frequency) is the number of audio samples that are taken per second.
•	Broadband models are used for audio that is sampled at no less than 16 kHz, which IBM® recommends for responsive, real-time applications (for example, for live-speech applications). Broadband model converts audio recorded at higher sampling rates to 16 kHz.
•	Narrowband models are used for audio that is sampled at no less than 8 kHz, which is the rate that is typically used for telephonic audio. It converts the audio to 8 kHz.
The service supports both broadband and narrowband audio for most languages and formats. It automatically adjusts the sampling rate of your audio to match the model that you specify before it recognizes speech.
Bit rate: -  is the number of bits of data that is sent per second. The bit rate for an audio stream is measured in kilobits per second (kbps). The bit rate is calculated from the sampling rate and the number of bits stored per sample. For speech recognition, IBM® recommends that you record 16 bits per sample for your audio.
Compression: - is used by many audio formats to reduce the size of the audio data. Compression reduces the number of bits stored per sample and thus the bit rate. Some formats use no compression, but most use one of the basic types of compression:
•	Lossless compression reduces the size of the audio with no loss of quality, but the compression ratio is typically small.
•	Lossy compression reduces the size of the audio by as much as 10 times, but some data and some quality is irretrievably lost in the compression.
With the Speech to Text service, you can safely use lossy compression to maximize the amount of audio that you can send to the service with a recognition request. Because the dynamic range of the human voice is more limited than, say, music, speech can accommodate a bit rate that is much lower than other types of audio. For speech recognition, IBM® recommends that you use 16 bits per sample for your audio and employ a format that compresses the audio data.
Channels: -  indicate the number of streams of the recorded audio:
•	Monaural (or mono) audio has only a single channel.
•	Stereophonic (or stereo) audio typically has two channels.
The Speech to Text service accepts audio with a maximum of 16 channels. Because it uses only a single channel for speech recognition, the service down mixes audio with multiple channels to one-channel mono during transcoding.
Notes about audio formats
•	For the audio/l16 format, you must specify the number of channels if your audio has more than one channel.
•	For the audio/wav format, the service accepts audio with a maximum of nine channels.
Audio Format: - Waveform Audio File Format (WAV) (audio/wav) is a container format that is often used for uncompressed audio streams, but it can contain compressed audio, as well. The service supports WAV audio that uses any encoding. It accepts WAV audio with a maximum of nine channels (due to an FFmpeg limitation).
We are using wav audio format. Required parvameters and Optional parameters for wav format are given below.
![image](https://user-images.githubusercontent.com/71966691/99433040-03dfc680-2933-11eb-8cb0-594f25056a80.png)
#### Output features
Word timestamps (timestamps=True/False):
The timestamps parameter tells the service whether to produce timestamps for the words it transcribes. By default, the service reports no timestamps. Setting timestamps to true directs the service to report the beginning and ending time in seconds for each word relative to the start of the audio.
End of phrase silence time (end_of_phrase_silence_time=0.8 seconds):
The following example requests show the effect of using the end_of_phrase_silence_time parameter. The audio speaks the phrase "one two three four five six," with a one-second pause between the words "three" and "four." The speaker might be reading a six-digit number from an identification card, for example, and pause to confirm the number.
Split transcript at phrase end (split_transcript_at_phrase_end=True/False):
The split_transcript_at_phrase_end parameter directs the service to split the transcript into multiple final results based on semantic features of the input. Setting the parameter to true causes the service to split the transcript at the conclusion of meaningful phrases such as sentences. The service bases its understanding of semantic features on the base language model that you use with the request along with a custom language model or grammar that you use. Custom language models and grammars can influence how and where the service splits a transcript.
### Note: - There are several other parameters available which can be used as per the problem in hand. 
Reference link: https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-output

Advantages of using IBM Watson: -
•	The transcription quality is good (in our use case)
•	Number of parameters are available to fine tune the pre trained models.
•	Few tasks(like audio frame rate adjustment and compression) are automatically done by the library.
•	Delimitation can be obtained (needed in our case).

Limitation: - This Is a paid software (Free trail is available which is good enough to tune the model).

### 3.9 Important Points
•	Confidence level of each extracted text line is used to refer the transcription quality.
•	FuzzyWuzzy Fuzzy string matching like a boss. It uses Levenshtein Distance to calculate the differences between sequences in a simple-to-use package.
  fuzz.token_set_ratio, fuzz.token_sort_ratio, fuzz.ratio and fuzz.partial_ratio
  
### Task3_Result
![image](https://user-images.githubusercontent.com/71966691/99433930-30e0a900-2934-11eb-8b7f-a8b59762bff7.png)


## Task_4: - Object Detection

### 4.1 Problem Statement
We need to detect the object in ad-video. We can perform this activity from a video or from extracted video frames. Here object detection is done directly from video. 

### 4.2 Objective
With objection detection we want to understand is there any influence of a particular object or any person in the ad-video.
The key benefit we get using object detection are:
3.	Influence of any high profile person
4.	Importance of any product 
5.	Importance of theme and color background used 

### 4.3 Literature Review
Object detection and tracking is one of the critical areas of research due to routine change in motion of object and variation in scene size, occlusions, appearance variations, and ego-motion and illumination changes. Specifically, feature selection is the vital role in object tracking. It is related to many real time applications like vehicle perception, video surveillance and so on. In order to overcome the issue of detection, tracking related to object movement and appearance. Most of the algorithm focuses on the tracking algorithm to smoothen the video sequence. On the other hand, few methods use the prior available information about object shape, color, texture and so on. Recently live cameras also used for objection detection, and the result is very good.

### 4.4 Machine Learning Process Flow
![image](https://user-images.githubusercontent.com/71966691/99434185-80bf7000-2934-11eb-8b2f-fb153b30a881.png)

Video Sequence: The detection of an object in video sequence plays a significant role in many applications such as movie video, ad-video, live camera recorded video with time
Object Recognition: Object recognition is a general term to describe a collection of related computer vision tasks that involve identifying objects in video or images.
Object Localization: Locate the presence of objects in an image and indicate their location with a bounding box.
Object Detection: Locate the presence of objects with a bounding box and types or classes of the located objects in an image with detection probability.

ImageAI provides convenient, flexible and powerful methods to perform object detection on videos. The video object detection class provided supports Resnet. To perform video object detection, we have to first download resnet50_coco_best_v2.0.1.ImageAI now allows live-video detection with support for camera inputs. 
Using OpenCV's VideoCapture() function, you can load live-video streams from a device camera, cameras connected by cable or IP cameras, and parse it into ImageAI's detectObjectsFromVideo() and detectCustomObjectsFromVideo() functions. All features that are supported for detecting objects in a video file is also available for detecting objects in a camera's live-video feed.
#### Network Structure of ResNet50.
![image](https://user-images.githubusercontent.com/71966691/99434366-b5332c00-2934-11eb-8bf3-e3dc7d829c33.png)
Resnet: A residual neural network is an artificial neural network of a kind that builds on constructs known from pyramidal cells in the cerebral cortex. Residual neural networks do this by utilizing skip connections, or shortcuts to jump over some layers.
#### Network Structure of Tiny_yolo_v3.
![image](https://user-images.githubusercontent.com/71966691/99434605-fdeae500-2934-11eb-8ede-2e8a5468f644.png)
TinyYolo_V3: defined a variation of the YOLO architecture called Tiny-YOLO. The Tiny-YOLO architecture is approximately 442% faster than its larger big brothers, achieving upwards of 244 FPS on a single GPU.

### 4.5 Resources:
Software: - VideoObjectDetection, opencv,  resnet50_coco_best_v2.0.1, ImageAI

### 4.6 Potential Data Challenges and Risks
Difficult to detect all the objects in video.
Selecting the right confidence threshold value is also a challenge.

### 4.7 Detailed Plan of Work
•	Using OpenCV's  VideoCapture() function, upload the saved video. Parse it into ImageAI's detectObjectsFromVideo() and detectCustomObjectsFromVideo() functions.
•	Output is another video with bounded boxes and probabilities and also a dataframe with all detected object list along with confidence level and other details.

### 4.8 Pre-Processing Steps
•	Download resnet50_coco_best_v2.0.1 and yolo-tiny.h5, save the models in the same location where video file is present.

### 4.9 Machine Learning Modelling and Techniques Applied
Object detection uses the following machine learning algorithm:
•	It uses resnet50_coco_best_v2.0.1 to detect predefined objects under the module
• Then uses Fast R-CNN network for extracting features from the proposed regions and outputting the bounding box and class labels.
• Model used is CRNN (Convolutional Recurrent Neural Network). It further has three main components:
     a.	Feature Extraction: Feature using Google ResNet.
     b.	Sequence Labelling: Sequence Labelling is done to keep video frames in sequence and detect objects from each using Long Short Term Memory (LSTM) algorithm.

### Task4_Result:
![image](https://user-images.githubusercontent.com/71966691/99435762-97ff5d00-2936-11eb-938d-a23648f2c86c.png)
Above image shows the dected objects using RetinaNet.
![image](https://user-images.githubusercontent.com/71966691/99435803-ab122d00-2936-11eb-8c10-fb06f24e6a2d.png)
Above table shows the list of detected objects using TinyYOLOV3. 
Note: I have shown the output for different videos in above result.

## Other Usefule Terms:

Sampling Rate: Number of samples per second used to code the speech signal (usually 16000, i.e. 16 kHz for a bandwidth of 8 kHz). Telephone speech is sampled at 8 kHz. 16 kHz is generally regarded as sufficient for speech recognition and synthesis. The audio standards use sample rates of 44.1 kHz (Compact Disc) and 48 kHz (Digital Audio Tape). Note that signals must be filtered prior to sampling, and the maximum frequency that can be represented is half the sampling frequency. In practice a higher sample rate is used to allow for non-ideal filters.
Sampling Resolution: Number of bits used to code each signal sample. Speech is normally stored in 16 bits. Telephony quality speech is sampled at 8 kHz with a 12bit dynamic range (stored in 8 bits with a non-linear function, i.e. A-law or U-law). The dynamic range of the ear is about 20 bits.
Lexicon or pronunciation dictionary: A list of words with pronunciations. For a speech recognizer it includes all words known by the system, where each word has one or more pronunciations with associated probabilities
Phoneme: An abstract representation of the smallest phonetic unit in a language which conveys a distinction in meaning. For example, the sounds /d/ and /t/ are separate phonemes in English because they distinguish words such as do and to. To illustrate phoneme differences across languages, the two /u/-like vowels in the French words tu and tout are not distinct phonemes in English, whereas the two /i/-like vowels in the English words seat and sit are not distinct phonemes in French.
Pitch or F0: The pitch is the fundamental frequency of a speech signal. In practice, the pitch period can be obtained from the position of the maximum of the autocorrelation function of the signal.
Fast Fourier transform: - A fast Fourier transform is an algorithm that computes the discrete Fourier transform of a sequence, or its inverse. Fourier analysis converts a signal from its original domain to a representation in the frequency domain and vice versa.

## References
https://nanonets.com/blo
https://pypi.org/project/easyocr/
https://github.com/JaidedAI/EasyOCR
https://github.com/Uberi/speech_recognition
https://deepspeech.readthedocs.io/en/v0.9.1/
https://imageai.readthedocs.io/en/latest/video/ 
https://github.com/githubharald/CTCWordBeamSearch
https://en.wikipedia.org/wiki/Residual_neural_network
https://www.easyproject.cn/easyocr/en/index.jsp#readme
https://www.vocapia.com/speech-to-text-technology.html
https://link.springer.com/article/10.1007/s13735-012-0016-2
https://en.wikipedia.org/wiki/Optical_character_recognition
https://www.youtube.com/watch?v=NRAqIjXaZvw&list=LL&index=3
https://www.youtube.com/watch?v=vSwEbscJTKk&list=LL&index=9
https://www.youtube.com/watch?v=P9GLDezYVX4&list=LL&index=12
https://stackabuse.com/object-detection-with-imageai-in-python/
https://analyticsindiamag.com/top-8-algorithms-for-object-detection/ 
https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-summary
https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-models
https://machinelearningmastery.com/object-recognition-with-deep-learning/ 
https://www.kaggle.com/shivamb/objects-bounding-boxes-using-resnet50-imageai
https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md 
https://jonathan-hui.medium.com/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359
https://www.ftdocs.com/new-blog/2018/7/10/why-you-need-ocr#:~:text=OCR%20can%20make%20your%20life%20easier%20by%3A&text=Reduce%20or%20eliminate%20costly%20data,and%20dramatically%20reducing%20processing%20times
https://medium.com/@nandacoumar/optical-character-recognition-ocr-image-opencv-pytesseract-and-easyocr-62603ca4357#:~:text=Limitations%20of%20both%20Tesseract%20and%20EasyOCR%3A&text=If%20a%20document%20contains%20languages,distorted%20perspective%2C%20and%20complex%20background

