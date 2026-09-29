# Whiteboard Content Extraction: An Accessibility Tool for Lecture Video

CS 59300: Application of Deep Learning, Fall 2026

## Team

- Steven LaGoy, lagosm01@pfw.edu
- Dalton Lybarger, lybads01@pfw.edu
- Pranav Rao, raops01@pfw.edu

## Project Overview

We propose a deep learning-based application that extracts handwritten content from recorded lecture video and converts it into a machine-readable, accessible format. Rather than requiring a single clean image of a handwritten expression, the system accepts an entire lecture recording, identifies the whiteboard or blackboard within the frame, and reconstructs the board's content over time as it is written.

The system samples and segments frames to separate the writing surface from the instructor and background, produces clean keyframes tagged with timestamps, and detects and groups handwritten content on those keyframes. That content, which may include equations and prose sentences, is classified and converted to text using deep learning methods, then reconstructed into a structured document. The final output is exported as a Microsoft Word document and a PDF, and delivered through a Flask web application usable from a computer or a mobile device.

## Motivation

Instructors frequently write equations, derivations, and explanatory notes on physical or digital whiteboards during recorded lectures. That content is valuable to students and to accessibility services, but it is locked inside video and is not searchable, copyable, or readily convertible into accessible formats such as screen-reader-friendly text.

This project investigates whether a deep learning pipeline can reliably recover that written content directly from existing lecture recordings, without asking instructors to change how they teach or requiring students to scrub through video to find a moment on the board.

## Stakeholders and Clients

**Primary:** instructors and students who use recorded lectures. The tool must connect to existing classroom recording technology rather than requiring a new capture setup, since classroom cameras already capture the front of the room, including the writing surface. We are collecting feedback through a client survey of professors in the Math and Computer Science departments.

**Secondary:** the university's Disability Access Center. Their interest is in assistive technology and compliance with ADA accessibility guidelines, and their involvement is meant to keep the output genuinely usable for accessibility workflows rather than only convenient.

**Sponsor:** Dr. Chen, who is advising the project's focus on combining frame extraction with OCR and on producing ADA-compliant output.

## Project Scope

### Must-Have Functional Requirements

- Accept video input of up to 150 minutes in length.
- Detect and extract handwritten content from the video.
- Recognize both mathematical equations and prose sentences.
- Convert recognized content to text via OCR.
- Implement and compare multiple deep learning methods for image-to-text conversion: a custom CNN, a pretrained OCR model, and a multimodal LLM.
- Export the reconstructed content to Microsoft Word and PDF.
- Provide a usable interface for non-technical users, requiring no command-line interaction.

### Must-Have Non-Functional Requirements

- A simple workflow that does not require the user to run any code directly.
- Quantitative accuracy reporting, including accuracy, precision, recall, F1-score, confusion matrices, and word error rate (WER).
- Processing time no greater than one times the runtime of the input video.
- Attention to accessibility in the design of the output.
- Documentation sufficient to reproduce our results.

### Nice-to-Have Functional Requirements

- Export to LaTeX (.tex).
- Recognition of graphs and diagrams in addition to text and equations.
- A screen-reader-optimized output format.
- Timestamp-linked search within the recovered content.
- Tolerance for camera motion and support for multiple video sources.

### Nice-to-Have Non-Functional Requirements

- An output retention policy, such as storing outputs for 30 days.
- Maintainable code that supports future extension.
- Scalability to longer or higher-resolution videos.

## Deep Learning Approach

The recognition pipeline consists of the following stages:

- Sample frames from the input video and segment the whiteboard or blackboard from the instructor and background.
- Produce clean keyframes tagged with timestamps, so that content is only considered once it is stable across multiple frames.
- Detect and group handwritten content within each keyframe.
- Classify and convert the grouped content into text using deep learning methods, including equations and prose sentences.
- Reconstruct the recognized content into a structured text output, ordered and timestamped to match its appearance on the board.
- Export the structured output to Word and PDF through the web application.

A central part of the project is comparing deep learning approaches to this recognition step rather than committing to a single method up front. We plan to implement and evaluate at least a custom CNN-based classifier, a pretrained OCR model, and a multimodal LLM, using a modular pipeline design that allows each method to be swapped in and out for comparison.

## System Evaluation

### Recognition Accuracy

- Accuracy, precision, recall, F1-score, and confusion matrices.
- Comparison across the multiple deep learning methods under evaluation.

### Content-Extraction Quality

- Recall and precision of the board content recovered from video, as distinct from character-level recognition accuracy.
- Handling of content that is only fully legible or complete across multiple frames.

### End-to-End Performance

- Processing time relative to input video length.
- Responsiveness of timestamp-based search over the recovered content.

### User Acceptance

- Feedback from instructors, students, and Disability Access Center staff on the usability and usefulness of the output.

## User Interface

The application will be implemented as a Flask web application, usable from either a computer or a mobile device, so that a user can upload a lecture recording and receive a Word document and PDF without writing or running any code. The interface is intended to be simple enough for non-technical users, consistent with our non-functional usability requirement.

## Requirements Gathering

We refined our requirements through a formal requirements presentation, feedback from our sponsor Dr. Chen, and an ongoing client survey of Math and Computer Science faculty about their use of recorded lectures, how they currently reuse handwritten board content, and what accessible output would be most useful to them. We will continue to incorporate survey responses as they arrive.

## Next Steps: Now to Midterm Review

- End-to-end test: accept a video, sample frames, perform OCR on at least one keyframe, and export text, without the pipeline crashing.
- Gather test footage from real or representative lecture recordings.
- Fully evaluate one deep learning model on test footage, and begin evaluating a second model before the midterm review.
- Establish a GitHub workflow, including branch organization, pull request reviews, and an even division of work across the team.
- Build a UI shell for the web application.

## Expected Outcome

At the completion of the project, we expect to have a working Flask web application that accepts a lecture video and produces a timestamped, accessible reconstruction of its whiteboard content as a Word document and a PDF. Alongside the application itself, we expect to produce a clear, evaluated comparison of deep learning approaches to this recognition task, informed by feedback from instructors, students, and the Disability Access Center.# Handwritten Mathematical Expression Recognition

**CS 59300: Application of Deep Learning – Fall 2026**

**Team:**
- Steven LaGoy – lagosm01@pfw.edu
- Dalton Lybarger – lybads01@pfw.edu
- Pranav Rao – raops01@pfw.edu
- Om Singhal – singo01@pfw.edu

## Project Overview

We propose a deep learning-based application for recognizing handwritten mathematical expressions, including digits, letter variables, and mathematical operations. The intended use case is assisting professors who use digital or virtual whiteboards to convert handwritten mathematical content into LaTeX, a machine-readable format.

The application will allow a user to provide an image of a handwritten mathematical expression, or potentially draw the expression directly within the application. The system will preprocess the input, identify individual handwritten characters, classify those characters using a deep learning model, and reconstruct the characters into a mathematical expression. The resulting expression will be displayed in a digital format including plain text and LaTeX.

## Motivation

Instructors of math or math-adjacent courses frequently use handwritten notation when explaining concepts, particularly on digital or virtual whiteboards. This allows instructors to communicate mathematical ideas naturally; however, the resulting handwriting can be difficult to reuse in digital materials or convert into accessible formats.

Our project will investigate whether deep learning can be used to recognize a defined set of handwritten mathematical characters and reconstruct simple mathematical expressions. Rather than attempting to recognize every possible mathematical symbol or handwritten document, we will establish a manageable vocabulary of characters based on both available datasets and feedback from potential users.

Potential users and stakeholders include professors and faculty in mathematics who use digital whiteboards, computer science faculty who regularly use mathematical notation, and university personnel involved in accessibility or instructional technology.

## Project Scope

The initial version of the application will focus on a predefined set of handwritten mathematical characters. The exact vocabulary will be determined during the requirements-gathering phase, but may include:

- Digits: 0–9
- Common mathematical variables: a, b, c, x, y, z
- Operators: +, −, ×, ÷, =
- Parentheses: ( )
- Basic comparison operators: <, >, ≥, ≤
- Other basic notation such as decimal points and division/fraction notation

These character sets are mostly present in datasets of handwritten ASCII characters. Digits and variables are most accessible from existing datasets.

The project will initially focus on recognizing relatively simple mathematical expressions. More complex notation involving spatial relationships, such as exponents, square roots, fractions, integrals, and summations, will be considered potential extensions rather than requirements for the minimum viable application.

## Deep Learning Approach

We plan to investigate convolutional neural network (CNN) architectures for handwritten character classification. The initial model will be trained to classify individual handwritten characters into the selected vocabulary.

The overall recognition pipeline will consist of:

- Inputting a handwritten mathematical expression.
- Preprocessing the image.
- Detecting or segmenting individual characters.
- Classifying the characters using a CNN-based model.
- Determining the ordering of the recognized characters.
- Reconstructing the mathematical expression.
- Displaying the expression in a machine-readable format.

Existing handwritten-character datasets, including MNIST for handwritten digits, will be investigated as potential sources of training data. Additional datasets or collected samples may be required for alphabetic characters and mathematical symbols.

Data preprocessing may include resizing, grayscale conversion, normalization, thresholding, noise reduction, and data augmentation.

## Model Evaluation

An important component of the project will be evaluating the performance of the deep learning model.

We will investigate how the size and composition of the character vocabulary affects recognition performance and computational requirements. For example, we may compare a model recognizing digits only with models that additionally recognize variables and mathematical operators.

Model performance will be evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

We will also consider computational factors such as training time, inference time, and model size.

In addition to evaluating isolated character classification, we will evaluate the complete application on handwritten expressions to identify errors in character detection, classification, and expression reconstruction.

## User Interface

The final system will be implemented as a user-facing application rather than requiring users to execute code manually. The interface will provide a way to upload or draw a handwritten expression and display the resulting recognized expression.

A possible workflow is:

**Handwritten Input → Recognition → Digital Expression → LaTeX / Accessible Representation**

The interface will be designed to be simple and intuitive so that a user can demonstrate the recognition process without interacting directly with the underlying Python code or model.

## Client and Requirements Gathering

Before finalizing the character vocabulary and application requirements, we plan to consult potential users. In particular, we will seek feedback from mathematics and computer science faculty who use handwritten mathematical notation in their teaching.

We will ask potential users about:

- Their use of virtual or digital whiteboards
- The mathematical notation they commonly write
- How they currently convert handwritten material into digital formats
- Difficulties they encounter when doing so
- Which characters and symbols would be most useful to recognize
- Whether machine-readable output such as LaTeX would be useful in their workflow

We may also consult university accessibility or instructional technology personnel to better understand potential accessibility applications.

This feedback will help us determine an appropriate character vocabulary and prevent the project from becoming unnecessarily broad.

## Project Timeline

With 12 weeks remaining in the semester, we anticipate the following development process:

**Weeks 3–4:** Requirements gathering, potential client interviews, literature/background research, and definition of the initial character vocabulary.

**Weeks 4–5:** Dataset selection or collection and development of the preprocessing pipeline.

**Weeks 5–7:** Development and training of a baseline CNN character classifier.

*Weeks 7–9:* Model experimentation, hyperparameter tuning, and performance evaluation.

*Weeks 8–10:* Character detection/segmentation and reconstruction of simple expressions.

*Weeks 10–13:* Development of the user-facing application and integration with the trained model.

**Weeks 13–14:** End-to-end testing, performance evaluation, and refinement.

**Weeks 14–15:** Final testing, documentation, bug fixes, and preparation for the final demonstration.

## Expected Outcome

At the completion of the project, we expect to have a functional application capable of recognizing a predefined set of handwritten mathematical characters and reconstructing simple mathematical expressions.

The project will provide both a practical demonstration of deep learning for handwritten character recognition and an investigation into the tradeoffs involved in expanding the character vocabulary. The final application will demonstrate the potential of converting handwritten mathematical content into a machine-readable representation that can be more easily reused in digital and accessibility-oriented workflows.
