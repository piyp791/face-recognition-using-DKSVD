%install the ksvd amd omp box

%add ksvdbox and omp box to path%

clc;
addpath(genpath('.\ksvdbox'));  % add K-SVD box
addpath(genpath('.\OMPbox')); % add sparse coding algorithem OMP

%% training of the classifier

% Images used for the below experiment can be found in the link below
% http://vision.ucsd.edu/~iskwak/ExtYaleDatabase/ExtYaleB.html

% maximum estimated completion time: 2mins

% Creating projections of images into lower dimensions using randomface.
% parameter setting
dims = 504; % dimension of random-face feature descriptor
IMG_H = 192; % input image size
IMG_W = 168;

% generate the random matrix
randmatrix = randn(dims,IMG_H*IMG_W);
l2norms = sqrt(sum(randmatrix.*randmatrix,2)+eps);
randmatrix = randmatrix./repmat(l2norms,1,size(randmatrix,2));

%read images to generate test and train features
%image directory
%change this directory to images path%
outerdir = 'E:\FSL\sharingcode-LCKSVD\CroppedYale';
files = dir(outerdir);
trainfeaturesArr = [];
testfeaturesArr = [];

H_train = zeros(38, 1216);
H_test = zeros(38, 1198);

train_startindex = 1;
test_startindex = 1;
train_endindex = -1;
test_endindex = -1;

% the below loop will be used to output train features, test features and
% test and train lable matrices
for K=1:length(files)
    if files(K).isdir == 1
        foldername = files(K).name;
        %disp(foldername);
        
        if length(foldername)<3
            continue;
        end
        
        %read files from this folder
        currentdir = strcat(outerdir, '/', foldername);
        %disp(currentdir)
        
        %extractBetween not working. for now, lets make do with this.
        class_str = extractBetween(foldername, 6,7);
        classnum = str2double(class_str);
        %disp(classnum)
        
        %regex applied to filter out Ambient named images that are not
        %required in training/testing.
        images = dir([currentdir '/*P00A*.pgm']);
        num = size(images, 1);
        %disp(num);
        
        %pick 32 random numbers from the total number of images.
        rng(123);
        randomarray = randperm(num);
        %pick first 32 random indexes
        trainindexes = randomarray(1:32);
        %disp(trainindexes);
        testindexes = setdiff(randomarray, trainindexes);
        disp(length(trainindexes));
        disp(length(testindexes));
        %disp(testindexes);
        
        if classnum<14
            train_startindex = ((classnum-1)*length(trainindexes)) + 1;
            train_endindex = train_startindex +length(trainindexes)-1;
        else
            train_startindex = ((classnum-2)*length(trainindexes)) + 1;
            train_endindex = train_startindex +length(trainindexes)-1;
        end
        
        test_endindex = test_startindex + length(testindexes) -1;
        %disp(startindex)        
        
        for I=train_startindex: train_endindex
            if classnum<14
                H_train(classnum, I) = 1;
            else
                H_train(classnum-1, I) = 1;
            end  
        end
        
        for I=test_startindex: test_endindex
            if classnum<14
                H_test(classnum, I) = 1;
            else
                H_test(classnum-1, I) = 1;
            end  
        end
        test_startindex = test_endindex +1;
        
        for J=1:length(images)
            if images(J).isdir ~=1 && contains(images(J).name, 'Ambient') ~=1 && ismember(J, trainindexes)
                %disp(images(J).name)
 				img = imread(strcat(currentdir, '/', images(J).name));
 				feature = double(img(:));
 				randomfacefeature = randmatrix*feature;
 				trainfeaturesArr = horzcat(trainfeaturesArr,randomfacefeature);
            elseif images(J).isdir ~=1 && contains(images(J).name, 'Ambient') ~=1 && ismember(J, testindexes)
				%disp(images(J).name)
 				img = imread(strcat(currentdir, '/', images(J).name));
 				feature = double(img(:));
 				randomfacefeature = randmatrix*feature;
 				testfeaturesArr = horzcat(testfeaturesArr,randomfacefeature);
            end
        end
        
    end
end

% apply KSVD to obtain dictionary, classifier and sparse representation
% matrices
% once the KSVD is applied we need to retrieve the matrices D, W and T from
% the output of KSVD
sqrt_gamma = 2;
params.data = [trainfeaturesArr; sqrt_gamma * H_train];
params.Tdata = 30;
params.dictsize = 570;
params.iterations = 50;
[ksvd_dict, ksvd_Gamma, ksvd_err] = ksvd(params,'');

% Now we shall extract the D, W and T 

dksvd_dict = ksvd_dict(1: size(trainfeaturesArr, 1) , :);
dksvd_w = ksvd_dict(size(trainfeaturesArr, 1) +1: size(ksvd_dict, 1), :);

% Now we need to re-normalize the output matrices (borrowed from LC-KSVD code)

l2norms = sqrt(sum(dksvd_dict.*dksvd_dict,1)+eps);
%dksvd_dict_norm = dksvd_dict ./ repmat(l2norms,size(dksvd_dict,1),1);
D = dksvd_dict ./ repmat(l2norms,size(dksvd_dict,1),1);
dksvd_w_norm = dksvd_w ./ repmat(l2norms,size(dksvd_w,1),1);
%dksvd_w_final = dksvd_w_norm ./ sqrt_gamma;
W = dksvd_w_norm ./ sqrt_gamma;

% Now we will use these values to classify 
% we will use omp to compute the sparse representation matrix for the test
% images
sparsity = 30;
sparse_class  = omp(D'*testfeaturesArr, D'*D, sparsity);

pred = [];
for i = 1: size(sparse_class, 2)
    l = W * sparse_class(:, i);
    %disp(l);
    [M, I] = max(l);
    %disp(i);
    %disp(I);
    pred(i) = I;
end

true_class = [];
for j = 1: size(H_test, 2)
    [M1, I1] = max(H_test(:, j));
    %disp(I1);
    true_class(j) = I1;
end
   
accuracy_yale_data = ((sum(pred == true_class)) * 100)/ (size(H_test, 2));

%% Test on syntethic data

% Generate new synthetic data from random data
rng(123);
meanarr1 = [-0.234, -0.3, -3.2, -0.1, -0.02, -0.6, -0.76, -1.43, -4.33, -3.0];
meanarr2 = [0.234, 0.3, 3.2, 4.5, 2.4, 1.4, 0.76, 1.43, 4.33, 3.0];
vararr = [1.1681e-04, 2.4351e-02, 3.1683951e-03, 1.161e-05, 4.12e-01, 2.213e-02, 1.234e-01, 3.213e-02, 1.234e-01, 1.161e-05];
disp(size(meanarr1));
disp(size(meanarr2));
disp(size(vararr));

matrix1 = [];
matrix2 = [];
for I=1:300
    index = randperm(10, 1)
    vector1 = normrnd(meanarr1(index), vararr(index), [1,100]);
    vector1 = vector1(:);
    matrix1 = horzcat(matrix1, vector1);
    %matrix1 = matrix1 + -1*randn(size(matrix1)) +0.002;
    matrix1 = matrix1 + -1*randn(size(matrix1)) ;
    vector2 = normrnd(meanarr2(index), vararr(index), [1,100]);
    vector2 = vector2(:);
    matrix2 = horzcat(matrix2, vector2);
    %matrix2 = matrix2 + 3*randn(size(matrix2)) +0.0004;
    matrix2 = matrix2 + 3*randn(size(matrix2)) ;
end
% randomly choose index and create train_features and test_features

rng(123);
rand_vector1 = randperm(300, 200);
train_features1 = matrix1(:, rand_vector1);
rng(245);
rand_vector2 = randperm(300, 200);
train_features2 = matrix2(:, rand_vector2);
train_features = horzcat(train_features1, train_features2);

matrix_temp1 = matrix1;
matrix_temp1(:, rand_vector1) = [];
test_features1 = matrix_temp1;
matrix_temp2 = matrix2;
matrix_temp2(:, rand_vector2) = [];
test_features2 = matrix_temp2;
test_features = horzcat(matrix_temp1, matrix_temp2);

% create H_train and H_test matrices

H_train = [repmat(1,1,200), repmat(0,1,200); repmat(0,1,200), repmat(1,1,200)];
H_test = [repmat(1,1,100), repmat(0,1,100); repmat(0,1,100), repmat(1,1,100)];

% apply KSVD to obtain dictionary, classifier and sparse representation
% matrices
% once the KSVD is applied we need to retrieve the matrices D, W and T from
% the output of KSVD
sqrt_gamma = 2;
params.data = [train_features; sqrt_gamma * H_train];
params.Tdata = 30;
params.dictsize = 150;
params.iterations = 50;
[ksvd_dict, ksvd_Gamma, ksvd_err] = ksvd(params,'');

% Now we shall extract the D, W and T 

dksvd_dict = ksvd_dict(1: size(train_features, 1) , :);
dksvd_w = ksvd_dict(size(train_features, 1) +1: size(ksvd_dict, 1), :);

% Now we need to re-normalize the matrices

l2norms = sqrt(sum(dksvd_dict.*dksvd_dict,1)+eps);
%dksvd_dict_norm = dksvd_dict ./ repmat(l2norms,size(dksvd_dict,1),1);
D = dksvd_dict./ repmat(l2norms,size(dksvd_dict,1),1);
dksvd_w_norm = dksvd_w./ repmat(l2norms,size(dksvd_w,1),1);
%dksvd_w_final = dksvd_w_norm ./ sqrt_gamma;
W = dksvd_w_norm./ sqrt_gamma;

% Now we will use these values to classify 
% we use omp function from OMP-BOX to compute the sparse representation
% matrix for the test images
sparsity = 30;
sparse_class  = omp(D'*test_features, D'*D, sparsity);

pred = [];
for i = 1: size(sparse_class, 2)
    l = W * sparse_class(:, i);
    %disp(l);
    [M, I] = max(l);
    %disp(i);
    %disp(I);
    pred(i) = I;
end

true_class = [];
for j = 1: size(H_test, 2)
    [M1, I1] = max(H_test(:, j));
    %disp(I1);
    true_class(j) = I1;
end
   
accuracy_synthetic_data = ((sum(pred == true_class)))/ (size(H_test, 2));






