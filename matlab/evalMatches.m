% keypoints is an N by 5 matrix [x,y,size,angle(degrees),ID]
% matches is an N by K matrix

function [ps] = evalMatches(keypoints,matches,im)

N  = size(matches,1);
xc = size(im,2)/2;
yc = size(im,1)/2;

% The parameter space is three dimensional therefore the 3
ps = zeros(N,3);

for n = 1:N
    if matches(n) ~= 0
        x   = [keypoints(n,1); keypoints(matches(n),1)];
        y   = [keypoints(n,2); keypoints(matches(n),2)];
        phi = [keypoints(n,4); keypoints(matches(n),4)]*pi/180; % in rads
        
        phi_ax = sum(phi)/2;
        if phi_ax < 0               % it does not matter the direction of the angle
            phi_ax = pi + phi_ax;
        end
        
        r      = (sum(x)/2-xc)*sin(phi_ax)-(sum(y)/2-yc)*cos(phi_ax);
        t      = norm(x-y)*sin(atan((y(1)-y(2))/(x(1)-x(2))));
        
        ps(n,1) = phi_ax; % sin(phi_ax)*200;   %   phi_ax*180/pi;
        ps(n,2) = r;
        ps(n,3) = t;
    end
end

ps(isnan(ps)) = 0;

ps(:,1) = ps(:,1)/max(abs(ps(:,1)));
ps(:,2) = ps(:,2)/max(abs(ps(:,2)));
ps(:,3) = ps(:,3)/max(abs(ps(:,3)));