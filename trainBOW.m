load('IDLabels.mat')
load('pokevocab');

imds = imageDatastore('train');
imds.Labels = ID_gt;
% tbl = countEachLabel(imds)


% Set the ImageDatastore ReadFcn
imds.ReadFcn = @(filename)preprocess(filename);

% bag = bagOfFeatures(imds);
categoryClassifier = trainImageCategoryClassifier(imds, bag);
confMatrix = evaluate(categoryClassifier, imds);

% disp(size(imds));

% img = readimage(imds,1);
% 
% h = size(img,1);
% w = size(img,2);
% d = h*w;

% disp(img(1:10,1));

% img = reshape(img, [d 1]);

% disp(img(1:10));





