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
