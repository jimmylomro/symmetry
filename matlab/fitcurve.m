function ctr = fitcurve(idx, matches, kpts, maxDeg)

K = max(idx);

ctr = cell(K,1);

for k = 1:K

    e_best = 0;
    
    for i = 0:3
        
        % Init rotation matrices
        r  = [cos(i*pi/4) -sin(i*pi/4); sin(i*pi/4) cos(i*pi/4)];
        
        midx = idx == k & matches ~= 0;
        x   = r * (kpts(midx,1:2) + kpts(matches(midx),1:2))'/2;
        
        for j = 1:maxDeg
            
            p = polyfit(x(1,:),x(2,:),j);
            y = polyval(p,x(1,:));
            e = sum(abs(x(2,:) - y))/size(x,2);
            
            if e < e_best || (j == 1 && i == 0)

                x_ctr  = min(x(1,:)):max(x(1,:));
                y_ctr  = polyval(p,x_ctr);
                ctr{k} = ([1,-1;-1,1].*r)*[x_ctr;y_ctr];
                e_best = e;
            end
        end
    end
end