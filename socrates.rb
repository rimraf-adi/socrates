class Socrates < Formula
  desc "Socrates & Plato — Generator-Critic Agent System"
  homepage "https://github.com/yourusername/socrates"
  url "https://github.com/yourusername/socrates/releases/download/v0.1.0/socrates-macos"
  sha256 "REPLACE_WITH_SHA256"
  license "MIT"

  def install
    bin.install "socrates-macos" => "socrates"
  end

  test do
    system "#{bin}/socrates", "--help"
  end
end
