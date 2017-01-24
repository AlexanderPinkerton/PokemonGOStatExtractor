% Load training and test data using |imageDatastore|.
datadir = fullfile('dataset');
trainingSet = imageDatastore(datadir,'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Preprocess each image as it is loaded.
trainingSet.ReadFcn = @(filename)presvm(filename);





img = readimage(trainingSet, 210);



% Extract HOG features and HOG visualization
% [hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
% [hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);

cellSize = [4 4];
% Hog Length 1008
hogFeatureSize = length(hog_4x4);

% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');

for i = 1:numImages
    img = readimage(trainingSet, i);
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;


% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels);






