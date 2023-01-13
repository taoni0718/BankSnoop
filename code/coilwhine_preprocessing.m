[audioIn1, fs1] = audioread('belkin_incharging1.wav');
[audioIn2, fs2] = audioread('belkin_noncharging1.wav');

fs = 32000;

audioIn1 = audioIn1(1:320000);
audioIn2 = audioIn2(1:320000);

audioIn1 = audioIn1(fs*3:fs*4-1);
audioIn2 = audioIn2(fs*3:fs*4-1);

% Filter the signal
fc = 1000; % Make higher to hear higher frequencies.
% Design a Butterworth filter.
[b, a] = butter(6,fc/(fs/2));
%freqz(b,a)

audioIn1 = highpass(audioIn1, fc, fs);
audioIn2 = highpass(audioIn2, fc, fs);

% , 'FrequencyRange',[4e3, 2e4]

%audioSegs = [audioIn2; audioIn1];

%figure
%subplot(211);
melSpectrogram(audioIn1, fs, 'Window', hann(256, 'periodic'), 'OverlapLength', 32, 'FFTLength', 256, 'NumBands', 64, 'FrequencyRange',[1.1e4, 1.6e4]);
caxis([-150 -100])
set(gca, 'xtick', [], 'xticklabel', [], 'ytick', [], 'yticklabel', []);
%subplot(212);
%melSpectrogram(audioIn2, fs, 'Window', hann(256, 'periodic'), 'OverlapLength', 32, 'FFTLength', 256, 'NumBands', 64, 'FrequencyRange',[1.1e4, 1.6e4]);
%caxis([-150 -100])