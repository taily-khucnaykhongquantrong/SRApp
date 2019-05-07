function [ perceptual_score ] = evaluate( sr, hr )

addpath(genpath(fullfile(pwd, 'utils')));
% Number of pixels to shave off image borders when calcualting scores
shave_width = 4;

% Set verbose option
verbose = true;

scores = calc_scores(sr, hr, shave_width, verbose);
perceptual_score = (mean([scores.NIQE]) + (10 - mean([scores.Ma]))) / 2;

% Saving
% save('your_scores.mat','scores');

end

