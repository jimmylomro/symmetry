%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function PlotClusterinResult(im, kpts, matches, X, IDX)

    figure;
    h1 = subplot(1,2,1);
    hold on;
    axis square;
    grid on;

    h2 = subplot(1,2,2);
    hold on;
    imshow(im);
    axis image;
    
    k=max(IDX);

    Colors=hsv(k);

    Legends = {};
    for i=0:k
        Xi   = X(IDX==i,:);
        
        if i~=0
            Style = 'x';
            Color = Colors(i,:);
            Legends{end+1} = ['Cluster #' num2str(i)];
            
            Li  = IDX==i & matches ~= 0;

            for pl = 1:length(Li)
                if(Li(pl))
                    plot(h2,[kpts(pl,1);kpts(matches(pl),1)],[kpts(pl,2);kpts(matches(pl),2)],'.-','Color',Color,'MarkerSize',5,'LineWidth',3);
                end
            end
        else
            Style = 'o';
            Color = [0 0 0];
            if ~isempty(Xi)
                Legends{end+1} = 'Noise';
            end
        end
        if ~isempty(Xi)
            if size(Xi,2) == 3
                scatter3(h1,Xi(:,1),Xi(:,2),Xi(:,3),Style,'MarkerEdgeColor',Color);
            elseif size(Xi,2) == 2
                scatter(h1,Xi(:,1),Xi(:,2),Style,'MarkerEdgeColor',Color);
            end
        end
    end
    
%     subplot(1,1,1);
%     hold off;
%     grid on;
    title(h1,['DBSCAN (No. of detected clusters = ' num2str(k) ')']);
    %legend(Legends);
    %legend('Location', 'NorthEastOutside');
    
%     subplot(1,1,1);
%     hold off;
    title(h2,'Symmetryc matches');

end