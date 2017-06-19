function convert_SFDB09_to_mat(path, output_folder) % TODO mettere il default output_folder="./"
    % convert data from .SFDB09 to .mat format
    % the SFDB09 file format (Short FFT DataBase, 2009 specification) is developed by Sergio Frasca and Ornella Piccinni
    
    % to read the SFDB09 data, we need a function (written by Pia Astone) defined inside the Snag Matlab package (written by Sergio Frasca)
    % Snag is a Matlab data analysis toolbox oriented to gravitational-wave antenna data
    % Snag webpage: http://grwavsf.roma1.infn.it/snag/
    % version 2, released 12 May 2017
    % installation instructions:
    % http://grwavsf.roma1.infn.it/snag/Snag2_UG.pdf

    % 'path' can refer both to a single file or a whole data directory
    % TODO vedere se funziona pure sulle liste di files
    
    % define a function to convert a single SFDB file
    function convert_single_file(file_path, output_folder) % TODO mettere il default output_folder="./"
        % adapted for the original code written by Pia Astone
        file_identifier=fopen(file_path);
        % the pia_read_block_09 function that can be found inside the Snag Matlab package, written by Sergio Frasca
        % Snag webpage: http://grwavsf.roma1.infn.it/snag/
        header = {};
        periodogram = {};
        autoregressive_spectrum = {};
        fft_data = {};
        % every single SFDB09 file includes a slice of 100 FFT
        number_of_FFT = 100;
        for fft_index = 1:number_of_FFT
            [head tps sps sft]=pia_read_block_09(file_identifier); % TODO rendere parallela la funzione
            if head.eof % the final files have less fft inside them
                break
            end
            header = [header, head];
            periodogram = [periodogram, tps];
            autoregressive_spectrum = [autoregressive_spectrum, sps];
            fft_data = [fft_data, sft];
        end
        
        header = struct2table(cell2mat(header)); % TODO squeeze(header)
        
        %header.eof is here discarded because not useful anymore
        %header.sat_howmany nowadays isn't used anymore: it was a saturation flag used in the early Virgo
        
        % whenever possible, data are converted to single precision (float32): ready for GPU computing
        % some attributes/values are redundant. they are saved anyway
        
        endianess = header.endian(1); % TODO capire a che serve
        
        if header.detector(1) == 0
            detector = 'Nautilus';
        elseif header.detector(1) == 1
            detector = 'Virgo';
        elseif header.detector(1) == 2
            detector = 'LIGO Hanford';
        elseif header.detector(1) == 3
            detector = 'LIGO Livingston';
        end
        
        % the 3 components of the detector's position (in equatorial cartesian coordinates) evaluated at half the FFT time window
        position = single(cat(2, header.px_eq,header.py_eq,header.pz_eq));
        % the 3 components of the detectors's velocity (in equatorial cartesian coordinates) evaluated at half the FFT time window
        velocity = single(cat(2, header.vx_eq,header.vy_eq,header.vz_eq));
        
        % FFT starting time % TODO valutare float64
        gps_seconds = header.gps_sec;
        gps_nanoseconds = header.gps_nsec; % nanosecondi gps (da sommare ai secondi, dopo averli moltiplicati per 10^-9 % TODO vedere perché sono 0
        gps_time = gps_seconds + gps_nanoseconds * 1e-9; % TODO rivedere per sicurezza
        first_UTC_time = gps2utc(gps_time(1)); % gps2utc is another function included in the Snag package % TODO it would be better to have the year before the month
        gps_time = single(gps_time);
        
        if header.typ(1) == 1
            fft_interlaced = false;
        elseif header.typ(1) == 2
            fft_interlaced = true;
        end
        
        reduction_factor = header.red(1); % 128 (expresses how much the autoregressive spectrum is subsampled with respect to the FFT. so here the FFT is averaged for 128 time intervals)
        
        fft_lenght = header.tbase(1);
        
        fft_index = header.nfft;
        
        % FFT starting time (using Modified Julian Date) (computed using seconds and nanoseconds) % TODO controllare questo fatto
        mjd_time = single(header.mjdtime);
        
        scaling_factor = header.einstein(1); % 10^-20
        
        spare1 = header.spare1; % not used yet
        spare2 = header.spare2;
        spare3 = header.spare3;
        percentage_of_zeros = single(header.spare4);
        spare5 = header.spare5;
        spare6 = header.spare6;
        lenght_of_averaged_time_spectrum = header.lavesp(1); % lenght of the FFT divided in pieces by the reduction factor (128) % TODO
        scientific_segment = header.spare8; % non used anymore % TODO capire se lista o singolo valore oppure se lasciare spare8
        spare9 = header.spare9;
        
        % normalization factor for the power spectrum extimated from the square modulus of the FFT due to the data quantity (sqrt(dt/nfft)) % TODO
        normalization_factor = header.normd(1); % TODO scalare o vettore
        
        % corrective factor due to power loss caused by the FFT window
        window_normalization = header.normw(1); % TODO nome
        
        % sampling time used to obtain a given frequency band, subsampling the data
        subsampling_time = header.tsamplu(1); % TODO controllare
        
        frequency_resolution = header.deltanu(1); % 1/t_ftt
        
        % number of data labeled with some kind of warning flag (eg: non-science flag) % TODO forse non usato qui
        number_of_flags = header.n_flag; % TODO originariamente lista di -1
        
        % number of artificial zeros, used to fill every time hole in the FFT (eg: non-science data)
        number_of_zeros = header.n_zeroes;
        
        % window type used in the FF
        if header.wink(1) == 0
            window_type = 'none';
        elseif header.wink(1) == 1
            window_type = 'Hanning';
        elseif header.wink(1) == 2
            window_type = 'Hamming';
        elseif header.wink(1) == 3
            window_type = 'MAP'; % "Maria Alessandra Papa" time window, used at Ligo
        elseif header.wink(1) == 4
            window_type = 'Blackmann flatcos'; % TODO
        elseif header.wink(1) == 5
            window_type = 'flat top cosine edge'; % suggested value here. we cannot use Hamming due to spindown % TODO
        end
        
        % number of samples in half (unilateral) FFT
        unilateral_number_of_samples = header.nsamples(1); % TODO non conservando la frequenza % TODO vettore per il file monco finale? CHECK
        
        starting_fft_frequency = header.frinit(1);
        
        starting_fft_sample_index = header.firstfrind(1);
        % if the FFT do not start from frequency 0, it indicates the first frequency index
        % the index refers to the number of samples, not to frequency (in opposition to starting_fft_frequency)
        
        periodogram = single(cell2mat(periodogram));
        
        autoregressive_spectrum = single(cell2mat(autoregressive_spectrum));
        
        fft_data = single(cell2mat(fft_data)); % complex numbers: float32 + i*float32 = complex64
        % TODO senza single() è molto molto più lento a salvere i file su disco
        
        if output_folder(end) ~= '/'
            output_folder = strcat(output_folder, '/');
        end
        new_file_path = output_folder + string(first_UTC_time) + '.mat'; % TODO creare automaticamente le sottocartelle del path tramite le informazioni nel file, in modo da avere automaticamente tutto ordinato
        
        % TODO save in h5 with the latest version (per ora 'MATLAB 5.0 MAT-file', almeno in lettura) (use '-v7.3' flag ?)
        save(str2mat(new_file_path),...
            'endianess','detector','gps_time','fft_lenght',...
            'starting_fft_sample_index','unilateral_number_of_samples','reduction_factor',...
            'fft_interlaced','number_of_flags','scaling_factor','mjd_time','fft_index',...
            'window_type','normalization_factor','window_normalization','starting_fft_frequency',...
            'subsampling_time','frequency_resolution','velocity','position',...
            'lenght_of_averaged_time_spectrum','number_of_zeros','spare1','spare2',...
            'spare3','percentage_of_zeros','spare5','spare6','scientific_segment','spare9',...
            'periodogram', 'autoregressive_spectrum', 'fft_data');
    end
    
    % if we are dealing with a directory
    if isdir(path)
        data_dir = path;
        if data_dir(end) ~= '/'
            data_dir = strcat(data_dir, '/');
        end
        % find all .SFDB09 files in the data directory and its subdirectories
        s = dir(strcat(data_dir, '**/*.SFDB09'));
    % if we are instead dealing with a single file
    elseif ~isdir(path)
        file_path = path;
        s = dir(file_path);
    end
    
    % create the complete absolute paths
    folder_paths = string({s.folder});
    file_names = string({s.name});
    complete_file_paths = folder_paths + '/' + file_names;
    
    % convert all the data % TODO renderlo vettoriale e parallelo con parfor
    for file_path = complete_file_paths
        display(str2mat('Converting ' + file_path));
        convert_single_file(file_path, output_folder);
    end
    % the compressed .mat output file is 500 times smaller than the original one if the data are zeros (data missing or flagged)
end



