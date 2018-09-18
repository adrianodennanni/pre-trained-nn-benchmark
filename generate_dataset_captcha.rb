require 'parallel'
require 'rmagick'
require 'securerandom'

puts 'Generating captchas...'

# Generate the diretories
dir_path = "./captcha_dataset"
unless Dir.exist?(dir_path)
  Dir.mkdir(dir_path)
  Dir.mkdir("#{dir_path}/train")
  Dir.mkdir("#{dir_path}/validation")
  Dir.mkdir("#{dir_path}/test")
end

# Function for generate and save captchas
def save_random_captcha(text, dataset)
  image = Magick::Image.new(160, 80)
  image.format = "jpg"
  image.gravity = Magick::CenterGravity
  image.background_color = 'white'

  draw = Magick::Draw.new
  draw.annotate(image, image.columns, image.rows, 0, 0, text) {
    self.gravity = Magick::CenterGravity
    self.pointsize = 28
    self.fill = 'darkblue'
    self.stroke = 'transparent'
    self.rotation = rand(-20..20)
  }

  image = image.gaussian_blur(0.0, rand)
  image = image.implode((rand(1)) / 10.0)
  image = image.swirl(rand(15))
  image = image.radial_blur(rand(5))
  image = image.add_noise(Magick::ImpulseNoise)

  File.write("./captcha_dataset/#{dataset}/#{text}.jpg", image.to_blob)
end

# Start saving the captcha_dataset
texts = []
(1..35_000).each { texts << SecureRandom.alphanumeric(6).upcase }
Parallel.each(texts.uniq, in_processes: 30, progress: "Creating train captchas") { |text| save_random_captcha(text, 'train') }

texts = []
(1..2_000).each { texts << SecureRandom.alphanumeric(6).upcase }
Parallel.each(texts.uniq, in_processes: 30, progress: "Creating validation captchas") { |text| save_random_captcha(text, 'validation') }

texts = []
(1..2_000).each { texts << SecureRandom.alphanumeric(6).upcase }
Parallel.each(texts.uniq, in_processes: 30, progress: "Creating test captchas") { |text| save_random_captcha(text, 'test') }
