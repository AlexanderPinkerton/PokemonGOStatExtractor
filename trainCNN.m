categories = 1:201;

load('IDLabels.mat')
imds = imageDatastore('train');
imds.Labels = ID_gt;
% tbl = countEachLabel(imds)


%Import the AlexNet to fine-tune
convnet = helperImportMatConvNet('alexnet');

% fc8
convnet.Layers(21, 1).OutputSize = 201;
convnet.Layers(21, 1).Weights = zeros(201, 4096);
convnet.Layers(21, 1).Bias  = zeros(201, 1);

% out
convnet.Layers(23, 1).ClassNames  = 1:201;
convnet.Layers(23, 1).OutputSize  = 201;


% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)preprocess(filename);


featureLayer = 'fc7';
trainingFeatures = activations(convnet, imds, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = imds.Labels;

% Train multiclass SVM classifier using a fast linear solver, and set
% 'ObservationsIn' to 'columns' to match the arrangement used for training
% features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');





