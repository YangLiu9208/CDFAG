function [target_database ,source_database] = compute_feature( para )

img_dir=para.img_dir
data_dir=para.data_dir;
target_dataSet=para.target_dataSet;
source_dataSet=para.source_dataSet;

target_img_dir = fullfile(img_dir, target_dataSet);
source_img_dir = fullfile(img_dir, source_dataSet);
target_data_dir = fullfile(data_dir, target_dataSet);
source_data_dir = fullfile(data_dir, source_dataSet);

%% calculate sift features or retrieve the database directory
%if skip_cal_sift,
    %database = retr_database_dir(rt_data_dir);
    disp('==================================================');
    fprintf('Loading the feature...\n');
    disp('==================================================');
    target_database =retr_descriptors_dir(target_img_dir, target_data_dir);
    source_database =retr_descriptors_dir(source_img_dir, source_data_dir);
%else
%    database = CalculateSiftDescriptor(rt_img_dir, rt_data_dir, para.gridSpacing, para.patchSize, para.maxImSize, para.nrml_threshold);
%end;


end

