function out = preprocess(filename)

        img = imread(filename);
        height = size(img,1);
        width = size(img,2);

        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(img)
            img = cat(3,img,img,img);
%             img = rgb2gray(img);
        end

        % Resize the image as required for the CNN.
        pokecrop = img(round(height*.10):round(height*.45),round(width*.20):round(width*.80),:);        
        out = imresize(pokecrop, [250 250]);
        
       
end
