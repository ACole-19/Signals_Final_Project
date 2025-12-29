clear; close all; clc;

%READING THE FILE
infile = 'sample_c_note.wav';  

[x, fs] = audioread(infile);
if size(x,2) > 1
    % downmix to mono (simple average)
    x = mean(x, 2);
end

% Normalize to peak of 1.0 (avoid division by zero)
peakval = max(abs(x));
if peakval == 0
    error('Audio is silent.');
end
x = x / peakval;

%Centering the slice
sliceDur = 1.0;                 
Nslice = round(sliceDur * fs);  

Ns = length(x);
center = floor(Ns/2);
startIdx = center - floor(Nslice/2) + 1;
endIdx   = startIdx + Nslice - 1;

% % If slice extends beyond signal, pad with zeros
if startIdx < 1 || endIdx > Ns
    y = zeros(Nslice,1);
    srcStart = max(1, startIdx);
    srcEnd   = min(Ns, endIdx);
    destStart = srcStart - startIdx + 1;
    destEnd   = destStart + (srcEnd - srcStart);
    y(destStart:destEnd) = x(srcStart:srcEnd);
else
    y = x(startIdx:endIdx);
end

% Apply Hann window
w = hann(Nslice, 'periodic');
ywin = y .* w;

% Zero-pad for better frequency resolution
% Choose Nfft at least nextpow2(Nslice), but larger to get finer bins
Nfft = max(2^nextpow2(Nslice), 8192);
Y = fft(ywin, Nfft);
% only positive frequencies (real FFT)
half = floor(Nfft/2) + 1;
Ypos = Y(1:half);
mag = abs(Ypos);
% Convert to magnitude in dB (for plotting) while still using linear mag for peak detection
magdb = 20*log10(mag + eps);

% Frequency axis
f = (0:half-1) * (fs / Nfft);

% Limit to 0 - 4000 Hz for plotting & analysis
maxFreq = 4000;
imax = find(f <= maxFreq, 1, 'last');

% -------- Peak detection --------
% Basic thresholds
minPeakHeight = max(mag(1:imax)) * 0.05;   % peaks > 5% of max
minProminence = max(mag(1:imax)) * 0.02;   % small prominence

[peakVals, peakLocs] = findpeaks(mag(1:imax), 'MinPeakHeight', minPeakHeight, ...
                                                'MinPeakProminence', minProminence, ...
                                                'MinPeakDistance', round(20/(fs/Nfft))); 
% If no peaks found (very quiet), relax thresholds
if isempty(peakLocs)
    [peakVals, peakLocs] = findpeaks(mag(1:imax), 'MinPeakHeight', max(mag(1:imax))*0.01);
end

% Convert peakLocs (indices) to approximate frequencies (bin centers)
peakFreqs_bin = (peakLocs - 1) * (fs / Nfft);  % bin-center freq

% Sub-bin (parabolic) interpolation for better freq estimate
% function inline:
parabolic_interp = @(A, k) deal( ... 
    (k - 1) + 0.5*(A(k-1)-A(k+1))/(A(k-1)-2*A(k)+A(k+1)), ...
    20*log10(A(k) - ( (A(k-1)-A(k+1))^2 / (8*(A(k-1)-2*A(k)+A(k+1))) )) ...
    );

% compute refined frequency estimates
peakFreqs = zeros(size(peakLocs));
for i = 1:length(peakLocs)
    k = peakLocs(i);
    % ensure neighbors exist
    if k > 1 && k < length(mag)
        delta = 0.5*(mag(k-1)-mag(k+1)) / (mag(k-1)-2*mag(k)+mag(k+1));
        % corrected bin index:
        binidx = (k-1) + delta; % zero-based
        peakFreqs(i) = binidx * (fs / Nfft);
    else
        peakFreqs(i) = (k-1) * (fs / Nfft);
    end
end

% Sort peaks by frequency
[peakFreqs_sorted, sidx] = sort(peakFreqs);
peakVals_sorted = peakVals(sidx);

% ---------- Estimate fundamental (f0) ----------
% Heuristic: look for strong peak in low-frequency band (e.g. 40 - 1000 Hz)
lowBandMax = 1000; lowBandMin = 40;
lowIdx = find(peakFreqs_sorted >= lowBandMin & peakFreqs_sorted <= lowBandMax);

if ~isempty(lowIdx)
    % choose the strongest (largest magnitude) peak within low band
    [~, rel] = max(peakVals_sorted(lowIdx));
    f0_est = peakFreqs_sorted(lowIdx(rel));
else
    % fallback: choose lowest-frequency peak available (above 20 Hz)
    allIdx = find(peakFreqs_sorted >= 20);
    if ~isempty(allIdx)
        f0_est = peakFreqs_sorted(allIdx(1));
    else
        % extreme fallback: first found peak
        f0_est = peakFreqs_sorted(1);
    end
end

% Find first 3 harmonics (1x, 2x, 3x)
harmonics = zeros(1,3);
harmonics(1) = f0_est;
toleranceFrac = 0.08;   % 8% tolerance
maxSearchHz = maxFreq;
for n = 2:3
    target = n * f0_est;
    if target > maxSearchHz
        harmonics(n) = NaN;
        continue;
    end
    % find peak nearest to target
    [~, idxNearest] = min(abs(peakFreqs_sorted - target));
    if abs(peakFreqs_sorted(idxNearest) - target) <= max(toleranceFrac * target, 30) % at least 30Hz or 8%
        harmonics(n) = peakFreqs_sorted(idxNearest);
    else
        % If no close peak, still pick the nearest but mark it
        harmonics(n) = peakFreqs_sorted(idxNearest);
    end
end

% -------- Plotting --------
figure('Color','w','Position',[200 200 900 500]);
plot(f(1:imax), magdb(1:imax), 'LineWidth', 1.2);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Magnitude Spectrum (0 - 4000 Hz)');
grid on;
xlim([0 maxFreq]);

% Mark all detected peaks
hold on;
plot(peakFreqs, 20*log10(peakVals + eps), 'rv', 'MarkerFaceColor','r');

% Label fundamental and two harmonics with big markers and text
markerStyles = {'ro','go','mo'}; % 1st f0 red, 2nd green, 3rd magenta
for k = 1:3
    if ~isnan(harmonics(k))
        % find where this harmonic corresponds in peak list (closest)
        [~, id] = min(abs(peakFreqs_sorted - harmonics(k)));
        freqLabel = harmonics(k);
        % plot emphasized marker
        plot(freqLabel, 20*log10(peakVals_sorted(id)+eps), markerStyles{k}, 'MarkerSize',10,'LineWidth',1.5);
        % add text label
        txt = sprintf('%d: %.2f Hz', k, freqLabel);
        text(freqLabel, 20*log10(peakVals_sorted(id)+eps)+3, txt, 'HorizontalAlignment','center', 'FontWeight','bold');
    end
end

% Save figure
saveas(gcf, 'spectrum_peaks.png');

% -------- Report results --------
fprintf('Estimated fundamental (f0): %.2f Hz\n', harmonics(1));
fprintf('First three harmonics (Hz):\n');
for k = 1:3
    if isnan(harmonics(k))
        fprintf('  Harmonic %d: (not in range)\n', k);
    else
        fprintf('  Harmonic %d: %.2f Hz\n', k, harmonics(k));
    end
end
fprintf('Figure saved as spectrum_peaks.png\n');


%% ============================
%           Part 2
% =============================

% 1. Compute period T0 and samples-per-period Nper
T0   = 1 / harmonics(1);          % f0 from Part 1
Nper = round(T0 * fs);

fprintf('Estimated period T0 = %.6f sec, Nper = %d samples\n', T0, Nper);

% 2. Extract exactly one period from the windowed slice
if Nper > length(ywin)
    error('Estimated period exceeds slice length; f0 may be too small.');
end

onePeriod = ywin(1:Nper);

% 3. Loop it (tile 20 times)
numTiles = 20;
noteLoop = repmat(onePeriod, numTiles, 1);

% 4. Save audio
outfile = 'note_loop.wav';
audiowrite(outfile, noteLoop, fs);
fprintf('Saved periodic audio as %s\n', outfile);

% --- Automatically play the audio ---
try
    % MATLAB built-in playback
    player = audioplayer(noteLoop, fs);
    play(player);
    fprintf('Playing note_loop.wav ...\n');
catch ME
    warning('Could not auto-play audio: %s', ME.message);
end

% 5. Confirm periodicity
fprintf(['The audio sounds periodic because it is created by repeating a single cycle\n' ...
         'of the waveform. Looping exactly one period guarantees a perfectly periodic\n' ...
         'signal, which makes Fourier Series analysis exact and straightforward.\n']);
%% ============================
%           Part 3
% ============================

% Use the onePeriod signal and sampling rate fs from earlier parts
x_per = onePeriod(:);         % ensure column vector
L     = length(x_per);        % samples in one period
T0    = L / fs;               % period duration
N     = 12;                   % number of harmonics

% Time axis for the single period
t = (0:L-1)' / fs;

%% 1. Compute a0, ak, bk for k = 1..N
% Continuous-time Fourier Series approximated by summation.
% a0 term:
a0 = (2/L) * sum(x_per);      % matches CTFS scaling for numerical data

ak = zeros(1, N);
bk = zeros(1, N);

for k = 1:N
    ak(k) = (2/L) * sum( x_per .* cos(2*pi*k*t/T0) );
    bk(k) = (2/L) * sum( x_per .* sin(2*pi*k*t/T0) );
end

fprintf('The coefficients are as follows for a0, ak, and bk:\n');
fprintf('a0 = %.6f\n', a0);

fprintf('ak values:\n');
fprintf('  %.6f', ak); 
fprintf('\n');

fprintf('bk values:\n');
fprintf('  %.6f', bk);
fprintf('\n');


%% 2. Convert to exponential form Ck (k = -N ... N)
% Relationship:
%   C0 = a0/2
%   Ck = (ak - j*bk)/2 for k>0
%   C-k = (ak + j*bk)/2 for k>0

kvals = -N:N;         % index range
Ck = zeros(size(kvals));

for idx = 1:length(kvals)
    k = kvals(idx);

    if k == 0
        Ck(idx) = a0/2;
    elseif k > 0
        Ck(idx) = 0.5*( ak(k) - 1j*bk(k) );
    else % k < 0 → use symmetry
        Ck(idx) = 0.5*( ak(-k) + 1j*bk(-k) );
    end
end

%% 3. Plot |Ck| and phase(Ck)
% Magnitude plot
figure('Color','w');
stem(kvals, abs(Ck), 'LineWidth', 1.4, 'Marker', 'o');
xlabel('Harmonic index k');
ylabel('|C_k|');
title('Exponential Fourier Series Magnitude Spectrum');
grid on;
saveas(gcf, 'note_efs_spectrum.png');

% Phase plot (unwrapped)
figure('Color','w');
plot(kvals, unwrap(angle(Ck)), 'o-', 'LineWidth', 1.4);
xlabel('Harmonic index k');
ylabel('Phase (radians)');
title('Exponential Fourier Series Phase Spectrum (Unwrapped)');
grid on;
saveas(gcf, 'note_efs_spectrum_phase.png');

fprintf('Saved figures: note_efs_spectrum.png and note_efs_spectrum_phase.png\n');

%% ============================
%           Part 4
% ============================

% We use:
%   - Ck (exponential FS coefficients) from Part 3
%   - kvals = -N:N
%   - noteLoop (the looped periodic signal) from Part 2
%   - fs (sampling rate)
%   - T0 (period) and N (number of harmonics)

% Time axis for the looped note
Lloop = length(noteLoop);
t_loop = (0:Lloop-1)' / fs;

%% 1. Reconstruct y(t) from exponential Fourier series partial sum
y_recon = zeros(size(t_loop));

for idx = 1:length(kvals)
    k = kvals(idx);
    y_recon = y_recon + Ck(idx) * exp(1j * 2*pi*k * t_loop / T0);
end

% Take real part (imaginary part is numerical noise)
y_recon = real(y_recon);

%% 2. Overlay reconstruction and looped note for first 2000 samples
Nplot = min(2000, Lloop);

figure('Color','w');
plot(1:Nplot, noteLoop(1:Nplot), 'k', 'LineWidth', 1.1); hold on;
plot(1:Nplot, y_recon(1:Nplot), 'r--', 'LineWidth', 1.3);
legend('Original Looped Note', 'TFS Reconstruction');
xlabel('Sample Index');
ylabel('Amplitude');
title('TFS Reconstruction vs. Looped Note (First 2000 Samples)');
grid on;

%% 3. Save figure
saveas(gcf, 'note_tfs_recon.png');
fprintf('Saved note_tfs_recon.png\n');

%% ============================
%           Part 5
% ============================

% Choose how many partials to synthesize
K = 5;        % you may change to 3–5 depending on assignment

% Frequency of fundamental:
f0 = harmonics(1);

% Time axis to match noteLoop
t_long = (0:length(noteLoop)-1)' / fs;

% Storage for summation
partials_sum = zeros(size(t_long));

fprintf('Generating partials...\n');

for k = 1:K
    
    % Find exponential coefficient C_k (positive harmonic)
    % (From Part 3: C_k = (a_k - j b_k) / 2)
    Ck_pos = 0.5*(ak(k) - 1j*bk(k));
    
    % Synthesize k-th partial
    % The complex sinusoid:  C_k e^{j k ω0 t} + C_-k e^{-j k ω0 t}
    % But for sinusoidal audio, this reduces to:
    partial_k = ak(k)*cos(2*pi*k*f0*t_long) + bk(k)*sin(2*pi*k*f0*t_long);
    
    % Normalize lightly to avoid clipping
    partial_k = partial_k / max(abs(partial_k) + eps);
    
    % Save this partial
    filename = sprintf('partial_k%d.wav', k);
    audiowrite(filename, partial_k, fs);
    fprintf('Saved %s\n', filename);
    
    % Add to running sum
    partials_sum = partials_sum + partial_k;
end

% Normalize the summed partials
partials_sum = partials_sum / max(abs(partials_sum) + eps);

% Save partials_sum.wav
audiowrite('partials_sum.wav', partials_sum, fs);
fprintf('Saved summed partials: partials_sum.wav\n');
