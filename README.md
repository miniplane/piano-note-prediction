# piano note prediction

This is a program to automatically detect and classify notes from a piano sound recording.

Due to github's size limit on repositories, it wasn't possible to upload the complete training data set.


The scripts are meant to run on a gpu, otherwise the program won't terminate within a reasonable time frame.

	I recommend installing anaconda, and using conda as a package manager and to create a virtual environment. 
	
	Create a virtual environment:
		conda create -n test_env

	Enter the virtual environment:
		conda activate test_env

	Install numpy and matplotlib with conda:
		conda install numpy
		conda install matplotlib

	Since installing cuda, keras, and tensorflow is nontrivial and differs from setup to setup, I recommend following a suitable guide or tutorial.

	Cuda can be downloaded from the nvidia homepage:
		https://developer.nvidia.com/cuda-downloads

	Install compatible versions of Tensorflow and Keras for gpu:
		https://www.tensorflow.org/install
		https://keras.io/




Running the software


	Train the model for note prediction:
		python3 train.py


	To apply the trained model to predict notes, run pitch_detection.py with the file name of a 24bit mono WAV piano recording as an argument. There are a few example files to test with in the "examples" folder. 

	An example call to predict notes could look like this:
		python pitch_detection.py examples/ode_an_die_freude.wav


	Note that while the visual representation of the predictions works better for short sound files, all predictions are printed to the terminal.


	Detect note onsents and plot the results:
		python onset_detection.py <somesong>


