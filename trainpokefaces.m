clear; clc; close all;

img_path = './train/';
img_dir = dir([img_path,'*CP*']);
img_num = length(img_dir);

% ---------------------GENERATE POKEFACES MATRIX----------------------
% ID_gt = zeros(img_num,1);
% CP_gt = zeros(img_num,1);
% HP_gt = zeros(img_num,1);
% stardust_gt = zeros(img_num,1);
% ID = zeros(img_num,1);
% CP = zeros(img_num,1);
% HP = zeros(img_num,1);
% stardust = zeros(img_num,1);
%
% % 2500 becauase images have be resized to 50x50
% % pokefaces = zeros(img_num,2500);
% % pokefaces = uint8(zeros(50,50,img_num));
% pokefacesC = uint8(zeros(50*50,img_num));
%
% load('pokemean');
%
% pokemeanC = reshape(mean, [50*50 1]);
%
% for i = 1:img_num
%
%     close all;
%     img = imread([img_path,img_dir(i).name]);
%     disp(i);
%     height = size(img,1);
%     width = size(img,2);
%
%     % Some images may be grayscale. Replicate the image 3 times to
%     % create an RGB image.
%     if ~ismatrix(img)
%         img = rgb2gray(img);
%     end
%
%     % Resize the image to 50x50.
%     pokecrop = img(round(height*.10):round(height*.45),round(width*.20):round(width*.80));
%     pokecrop = imresize(pokecrop, [50 50]);
%     pokecropC = reshape(pokecrop, [50*50 1]);
%     pokefacesC(:,i) = pokecropC - pokemeanC;
%
% end

% -----------------------------------------------------------------------

% ----------GENERATE POKEFACE MEAN--------------
% load('pokefaces');
% mean = zeros(50,50);
% for i = 1:img_num
%     mean = mean + im2double(pokefaces(:,:,i));
% end
%
% mean = mean / img_num;
% mean = im2uint8(mean);
% imshow(mean);
% pause();
% ---------------------------------------------

load('pokefacesC');
load('pokemeanC');
load('pokemean');
load('IDLabels');
% imshow(pokefaces(:,:,1) - mean);

% V U S
[coefficients ,eigenvectors,eigenvalues] = pca(im2double(pokefacesC));

% Visualize the eigenvectors for debugging
% for i=1:size(eigenvectors,2)
%    
%     imshow(reshape(eigenvectors(:,i),[50 50]));
%     pause();
%     
%     
% end






img = imread('val/070_CP23_HP13_SD200_0149_37.jpg');
height = size(img,1);
width = size(img,2);

% Some images may be grayscale. Replicate the image 3 times to
% create an RGB image.
if ~ismatrix(img)
    img = rgb2gray(img);
end

% Resize the image to 50x50.
pokecrop = img(round(height*.10):round(height*.45),round(width*.20):round(width*.80));
pokecrop = imresize(pokecrop, [50 50]);
pokecropC = reshape(pokecrop, [50*50 1]);
pokecropC = pokecropC - pokemeanC;

% imshow(pokecrop);
% pause();
% imshow(pokecrop - mean);
% pause();

% Center the image with the training mean
centered = pokecropC - pokemeanC;

% Project new image into eigenspace
imgcoeff = eigenvectors' * im2double(centered);

% Find label corresponding to closest coefficient in training set.


load('pokeclassifier3');
% pokeclassifier = fitcecoc(coefficients', ID_gt);






label = predict(pokeclassifier, imgcoeff');
label = ID_gt(label);
disp(label);






% distances = zeros(2500,2);
% for coeffindx=1:size(coefficients,1)
%     disp(coeffindx);
% %     diff = imgcoeff - coefficients(coeffindx,:);
% %     distance = sqrt(diff * diff');
%     distance = norm(imgcoeff - coefficients(coeffindx,:));
%
%     %Store the distance to each feature and its class
%     distances(coeffindx, 1) = distance;
%     distances(coeffindx, 2) = ID_gt(coeffindx);
% end
%
% sortedDistances = sortrows(distances,1);
% predict_label = mode(sortedDistances(1:5,2),1);






