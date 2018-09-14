require 'CSV'
require 'down' # Easy file download
require 'fileutils'
require 'parallel' # Easy multi-thread processing
require 'rmagick' # Image manipulation
require 'ruby-progressbar'

# Labels' categories
categories_labels = {
  bird: '/m/015p6',
  cat:  '/m/01yrx',
  dog:  '/m/0bt9lr'
}


# URL for .csv files containing images ids and URLs
  csv_files = {
  label_id_train:      'https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels.csv',
  label_id_validation: 'https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels.csv',
  label_id_test:       'https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels.csv',
  id_url_train:        'https://storage.googleapis.com/openimages/2018_04/train/train-images-with-labels-with-rotation.csv',
  id_url_validation:   'https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv',
  id_url_test:         'https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv'
}


# Download CSV files
puts 'Downloading CSV files...'
download_path = './csv_files_download'
unless File.directory?(download_path)
  FileUtils.mkdir_p(download_path)
end

csv_files.each do |k, url|
  progress_bar = nil
  next if File.file?("#{download_path}/#{k}.csv")
  Down.download(
    url,
    destination: "#{download_path}/#{k}.csv",
    content_length_proc: Proc.new {|l| progress_bar = ProgressBar.create(total: l, format: '%a |%b>>%i| %p%% %t', title: "Downloading #{k}.csv") },
    progress_proc:       Proc.new {|x| progress_bar.progress = x }
  )
end
puts 'CSV files downloaded.'

# Get the ids from our desired categories
puts 'Getting images ids...'
desired_labels = categories_labels.values
ids = {}
Dir["#{download_path}/label_id_*"].each do |csv_file|
  total_lines = `wc -l "#{csv_file}"`.strip.split(' ')[0].to_i
  progress_bar = ProgressBar.create(total: total_lines, format: '%a |%b>>%i| %p%% %t', title: "Getting #{csv_file} ids")
  dataset = csv_file[/(train|test|validation)/]
  CSV.foreach(csv_file, headers: true) do |row|
    progress_bar.increment
    ids[row['ImageID']] = [row['LabelName'], dataset] if(row['Confidence'] == '1' && desired_labels.include?(row['LabelName']))
  end
end
puts 'Got all image ids.'

# Get the URLs from our ids
puts 'Getting images urls...'
urls = {}
desired_labels.each{ |label| urls[label] = { 'train' => [], 'validation' => [], 'test' => [] } }
Dir["#{download_path}/id_url_*"].each do |csv_file|
  total_lines = `wc -l "#{csv_file}"`.strip.split(' ')[0].to_i
  progress_bar = ProgressBar.create(total: total_lines, format: '%a |%b>>%i| %p%% %t', title: "Getting #{csv_file} urls")
  dataset = csv_file[/(train|test|validation)/]
  CSV.foreach(csv_file, headers: true) do |row|
    progress_bar.increment
    if ids.key?(row['ImageID'])
      urls[ids[row['ImageID']][0]][ids[row['ImageID']][1]] << row['Thumbnail300KURL'] unless row['Thumbnail300KURL'].to_s == ''
    end
  end
end
puts 'Got all image URLs.'


# Download the images
puts 'Downloading images...'
dir_path = "./dataset"
unless Dir.exist?(dir_path)
  Dir.mkdir(dir_path)
  categories_labels.keys.each do |category|
    Dir.mkdir("#{dir_path}/#{category}")
    Dir.mkdir("#{dir_path}/#{category}/train")
    Dir.mkdir("#{dir_path}/#{category}/validation")
    Dir.mkdir("#{dir_path}/#{category}/test")
  end
end

fixed_lenght = 100.0

files_count = Hash.new(0)

urls.each do |label, datasets|
  category = categories_labels.key(label)
  datasets.each do |dataset, urls|
    Parallel.each(urls, progress: "Downloading and resizing pictures: #{category}, #{dataset}", in_threads: 20) do |url|
      temp_file = Down.open(url, max_redirects: 0)

      next unless temp_file.data[:status].to_i == 200
      image = Magick::Image::from_blob(temp_file.read).first
      smaller_dimension = [image.columns, image.rows].min
      ratio = fixed_lenght/smaller_dimension
      image = image.scale(ratio)
      image.write("#{dir_path}/#{category}/#{dataset}/#{url[/\w+_\w+/]}.jpg")
      files_count["#{category}/#{dataset}"] += 1
    rescue StandardError
    end
  end
end

puts 'All images downloaded.'
files_count.each do |dataset, count|
  puts "#{dataset} has #{count.to_s} files."
end
