% Symmetry detection using c++ libraries

im=imread('../res/img1.jpg');
im=imresize(im,400/max(size(im)));

% Detection parameters
threshold   = 0.5;      % non maxima supression threshold
percent     = 0.04;     % kernSize in percent of image width
nMaps       = 10;       % number of feature maps

% Description parameters
pattScale   = 1;        % size of the pattern

% Matching parameters
matchDist = 120;        % hamming distance
maskFeats = true;       % match only keypoints of the same class

% DBSCAN parameters
epsilon     = 0.05;     % dbscan epsilon
minPts      = 50;       % dbscan minimum points

% Axis regression
singleAx    = false;    % single symmetry axis detection
fitMaxDeg   = 3;        % maximum degree for regression


% Initialize interface          
symmetry('init', ...
    'own_threshold',threshold,'own_kernSize',round(max(size(im))*percent),'own_nMaps',nMaps, ... % detection parameters
    'symbrisk_patternScale', pattScale, ...
    'matcher_dist', matchDist, 'matcher_maskFeats', maskFeats, ...
    'dbscan_epsilon', epsilon, 'dbscan_minPts', minPts);


% Load image
symmetry('loadImage',im);

% measure time
tic;

% detection
keypoints = symmetry('detect');


% description
[keypoints1, descriptors, descriptorsM] = symmetry('describe');


% mirror matching
matches = symmetry('knnMatch', descriptors, descriptorsM);


% parameter space
ps = evalMatches(keypoints1,matches,im);


% clustering
maskIdx = find(matches ~= 0);
idxSuc = symmetry('cluster', ps(maskIdx,1:2));
idx = zeros(size(matches));
idx(maskIdx) = idxSuc;


% free memory
symmetry('terminate');


% axis regression
if (singleAx)
    mx = 0;
    id = 1;
    for i = 1:max(idx)
        if(sum(idx == i) > mx)
            mx = sum(idx == i);
            id = i;
        end
    end
    idx = idx == id;
end

ctr = fitcurve(idx, matches, keypoints1, fitMaxDeg);


% measure time
time = toc;


% display

PlotClusterinResult(im, keypoints1, matches, ps(:,1:2), idx);
figure;
imshow(im);
hold on;
K = size(ctr,1);
for k = 1:K
    plot(ctr{k}(1,:),ctr{k}(2,:),'y-','LineWidth',5);
end

for k = 1:K
    kidx = idx == k;
    for i=1:size(matches,1)
        if matches(i,1)~=0 && kidx(i)
            j=matches(i,1);
            plot((keypoints1(i,1)+keypoints1(matches(i,1),1))/2,...
                (keypoints1(i,2)+keypoints1(matches(i,1),2))/2,'b.','MarkerSize',5);
            plot([keypoints1(i,1);keypoints1(matches(i),1)], ...
                [keypoints1(i,2);keypoints1(matches(i),2)], 'r.','MarkerSize',5,'LineWidth',5);
        end
    end
end

disp(time);
