% Step 1: Initialize variables
numFrames = 1; % Number of frames to create
xRange = linspace(0, 2*pi, numFrames); % Range of x-values for the plot

% Step 2 & 3: Create the plot loop and set axis limits
F = cell(numFrames, 1); % Preallocate cell array for frames
for t = 1:numFrames
    figure; % Create a new figure for each frame
    plot(xRange(t), sin(xRange(t))); % Update the plot with the current iteration value
    xlim([0, 2*pi]); % Set x-axis limits
    ylim([-1, 1]); % Set y-axis limits
    F{t} = getframe(gcf); % Capture the frame
end

% Step 4: Create a VideoWriter object
videoFileName = 'dynamicPlot.avi';
videoWriter = VideoWriter(videoFileName);

% Step 5: Open the VideoWriter object
open(videoWriter);

% Step 6: Write frames to video
for t = 1:numFrames
    writeVideo(videoWriter, F{t});
end

% Step 7: Close the VideoWriter object
close(videoWriter);
