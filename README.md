# face-recognition-using-DKSVD

Setup instructions

1. Clone the git repository.
2. Unzip the ompbox10 and ksvdbox13 zip files.
3. Install ompbox using the README file inside the ompbox10 folder. 
4. Make sure it has been installed correctly by running the ompdemo.m file.
4. Install ksvdbox13 using the same procedure.
5. Make sure it has been correctly by running the file ksvddemo.m file.
7. Download the Extended Yale dataset from the following link:
    http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html
	
8. Configure the path of the dataset accordingly in the source file dskvd.m at line 30:

outerdir = [Your image path];

9. Run the script dskvd.m.

You can read the project details and theory in the Project_Report.pdf file.