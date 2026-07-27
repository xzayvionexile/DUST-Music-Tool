# DUST-Music-Tool
Music Tool using Python and librosa

Used Claude and Python to create a Music Sampling software. 
The program uses librosa library to parse through a folder of audio files and extract from it harmonic data and beat data. 
It then analyzes the harmonic data to obtain an understanding of what key the song is in. 
After, it analyzes the beat data to understand what BPM it is in. Once it has the two, 
it superimposes the beat data over the harmonic data to create an indexed audio file that is formatted to a BPM structure; 
making it easier to later reconstruct. This is the first process; 
initiated by running anaylze_and_slice.py and following the comments left in the code. 
The second process reconstructs all of this data into a usable harmonic ‘sample.’ 
It creates an audio file of your desired length in bars, desired key, 
and desired bpm to use in your own music creation by running python generate.py 
