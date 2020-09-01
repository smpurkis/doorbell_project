# doorbell project
*(Currently a work in progress)*

TLDR: A face recognition doorbell, that announces who is at the door as they approach.

This is a personal project I'm planning on working on in my free time.

Description: A face rocognition doorbell using from the front door, that announces who is at the door at they approach.
It uses motion detection to detect any moving person, then attempts to find any faces in the frames. If it finds any, it compares them
to a list of known faces. On a positive detection it announces that person's name. On a negative detection it announces
that someone is at the door.

It is currently setup as a client (monitoring the video input) and a server (handling the face detection matches and announcements). 
Plus a good chance to experiment with python server APIs.
