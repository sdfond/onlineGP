load('sonarData.mat');

ey = sigTest(:,end);
ty = sigTrain(:,end);
meany = mean(ty);
for i = -3:3
    hypf = sprintf('res%d.mat', i);
    load(hypf);

    % display the result
    z = 1:length(ey);
    z = z';

    % mean +- 2* standard variance (95% confidence interval)
    a = figure(i+4);
    f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 

    fill([z; flipdim(z,1)], f, [7 7 7]/8);
    hold on;
    plot(z, ey, 'bx', 'MarkerSize', 8, 'LineWidth', 2);
    xlabel('Test Data');
    ylabel('Prediction Result');
    plot(z, m, '-ro', 'MarkerSize', 9);
    plot(z, repmat(meany, length(ey), 1), '-g');
    legend('95% CI', 'real height', 'estimated height', 'curb mean', 'location', 'best');
    hold off;
    fname = sprintf('res%d.eps', i);
    print(fname, '-depsc');
    close(a);
end