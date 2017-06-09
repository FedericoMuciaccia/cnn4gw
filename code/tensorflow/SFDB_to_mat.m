function SFDB_to_mat(path)
    % convert data from .SFDB09 to .mat format
    % 'path' can be a single file or a whole data directory
    % TODO vedere se funziona pure sulle liste di files
    
    % define a function to convert a single SFDB file
    function convert_single_file(file_path)
        % adapted for the original code written by Pia Astone
        file_indentifier=fopen(file_path);
        % the pia_read_block_09 function that can be found inside the Snag Matlab package, written by Sergio Frasca
        % Snag webpage: http://grwavsf.roma1.infn.it/snag/
        header = {};
        periodogram = {};
        autoregressive_spectrum = {};
        data = {};
        % every single SFDB09 file includes a slice of 100 FFT
        number_of_FFT = 100;
        for fft_index = 1:number_of_FFT
            [head tps sps sft]=pia_read_block_09(file_indentifier); % TODO rendere parallela la funzione
            if head.eof % the final files have less fft inside them
                break
            end
            header = [header, head];
            periodogram = [periodogram, tps];
            autoregressive_spectrum = [autoregressive_spectrum, sps];
            data = [data, sft];
        end
        
        header = struct2table(cell2mat(header)); % TODO squeeze(header)
        
        %header.eof here discarded because not useful anymore
        %header.sat_howmany nowadays not used anymore: it was a saturation flag used in the early Virgo
        
        % whenever possible, data are converted to single precision (float32): ready for GPU computing
        
        endianess = header.endian(1); % TODO
        
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
        
        % FFT starting time % TODO combinarli in float32
        gps_seconds = header.gps_sec;
        gps_nanoseconds = header.gps_nsec; % nanosecondi gps (da sommare ai secondi, dopo averli moltiplicati per 10 alla -9 % TODO vedere perché sono 0
        
        if header.typ(1) == 1
            fft_interlaced = false; % TODO squeeze
        elseif header.typ(1) == 2
            fft_interlaced = true;
        end
        
        reduction_factor = header.red(1); % 128 di quando è sottocampionato lo spettro autoregressivo rispetto alla fft (fft mediata 128 volte) TODO
        
        fft_lenght = header.tbase(1);
        
        fft_index = header.nfft;
        
        %TODO size 1 1 per gli scalari: 1 riga e 1 colonna
        
        % FFT starting time (using Modified Julian Date) (computed using seconds and nanoseconds TODO CHECK) % TODO ridondante
        mjd_time = single(header.mjdtime);
        
        scaling_factor = single(header.einstein(1)); % 10^-20 % TODO Einstein
        
        spare1 = header.spare1; % not used yet
        spare2 = header.spare2;
        spare3 = header.spare3;
        percentage_of_zeros = single(header.spare4(1)); % TODO CHECK
        spare5 = header.spare5;
        spare6 = header.spare6;
        lenght_of_average_time_spectrum = header.lavesp(1); % lenght of the FFT divided in pieces by the reduction factor (128) % TODO parametro ridondante
        scientific_segment = header.spare8; % non used anymore % TODO capire se lista o singolo valore oppure se lasciare spare8
        spare9 = header.spare9;
        
        % normalization factor for the power spectrum extimated from the square modulus of the FFT due to the data quantity (sqrt(dt/nfft)) % TODO
        normalization_factor = single(header.normd(1)); % TODO scalare o lista
        
        % corrective factor due to power loss caused by the FFT window
        window_normalization = single(header.normw(1)); % TODO nome
        
        % sampling time used to obtain a given frequency band, subsampling the data
        subsampling_time = single(header.tsamplu(1)); % TODO controllare
        
        frequency_resolution = single(header.deltanu(1)); % 1/t_ftt
        
        % number of data labeled with some kind of warning flag (eg: non-science flag) % TODO forse non usato qui
        number_of_flags = header.n_flag(1); % TODO originariamente lista di -1
        
        % number of artificial zeros, used to fill every hole in the FFT (eg: non-science data)
        number_of_zeros = header.n_zeroes(1); % TODO CHECK
        
        % window type used in the FF
        if header.wink(1) == 0
            window_type = 'none';
        elseif header.wink(1) == 1
            window_type = 'Hanning';
        elseif header.wink(1) == 2
            window_type = 'Hamming';
        elseif header.wink(1) == 3
            window_type = 'MAP'; % TODO Maria Alessandra ... , usata a Ligo
        elseif header.wink(1) == 4
            window_type = 'Blackmann flatcos'; % TODO
        elseif header.wink(1) == 5
            window_type = 'flat top cosine edge'; % suggested value here. we cannot use Hamming due to spindown % TODO
        end
        
        % number of samples in half (unilateral) FFT
        unilateral_number_of_samples = header.nsamples(1); % TODO non conservando la frequenza
        
        starting_fft_frequency = single(header.frinit(1)); % TODO ridondante
        
        starting_fft_sample_index = header.firstfrind(1);
        % TODO se fft non a partire da frequenza 0 % first fequency index
        % TODO qui in numero di samples e non in frequenza
        
        periodogram = single(cell2mat(periodogram));
        
        autoregressive_spectrum = single(cell2mat(autoregressive_spectrum));
        
        data = single(cell2mat(data));
        
        % TODO save in h5 with the latest version (per ora 'MATLAB 5.0 MAT-file') (use '-v7.3' flag ?)
        save(strcat(file_path, '.mat'),...
            'endianess','detector','gps_seconds','gps_nanoseconds','fft_lenght',...
            'starting_fft_sample_index','unilateral_number_of_samples','reduction_factor',...
            'fft_interlaced','number_of_flags','scaling_factor','mjd_time','fft_index',...
            'window_type','normalization_factor','window_normalization','starting_fft_frequency',...
            'subsampling_time','frequency_resolution','velocity','position',...
            'lenght_of_average_time_spectrum','number_of_zeros','spare1','spare2',...
            'spare3','percentage_of_zeros','spare5','spare6','scientific_segment','spare9',...
            'periodogram', 'autoregressive_spectrum', 'data');
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
    
    % convert all the data % TODO renderlo vettoriale e parallelo
    for file_path = complete_file_paths
        display(str2mat('Converting ' + file_path));
        convert_single_file(file_path);
    end
    % the compressed .mat output file is 500 times smaller than the original one if the data are zeros (data missing or flagged)
end



