# Tweet Summarizer Pipeline

A complete AI-powered system that scrapes tweets from Twitter users, generates intelligent summaries using a fine-tuned language model, and scores the quality of summaries using a reward model.

## 🚀 Features

- **Twitter Scraping**: Automatically scrapes tweets from any Twitter user using Selenium
- **Intelligent Ranking**: Sorts tweets by engagement (likes + retweets + replies) to focus on the most popular content
- **AI Summarization**: Uses a fine-tuned Qwen 1.5 model with LoRA (Low-Rank Adaptation) for efficient summarization
- **Quality Scoring**: Employs a trained reward model to score summary quality
- **Multiple Interfaces**: Command-line tool, Python API, and Jupyter notebook
- **Comprehensive Output**: Provides detailed statistics, compression ratios, and engagement metrics

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Selenium
- Chrome browser and ChromeDriver
- CUDA (optional, for GPU acceleration)

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RLHFSummarise
```

2. Install dependencies:
```bash
pip install -r backend/requirements.txt
```

3. Make sure you have Chrome installed and ChromeDriver in your PATH

4. Ensure you have the required model files:
   - `lora_weights.pt` - Fine-tuned LoRA weights for the summarizer
   - `qwen_reward_model.pt` - Trained reward model for quality scoring

## 🎯 Quick Start

### Command Line Usage

```bash
# Basic usage
python run_tweet_summarizer.py elonmusk

# With options
python run_tweet_summarizer.py dril --count 15 --save --max-length 200

# Save results to file
python run_tweet_summarizer.py username --save --output results.json

# Use specific device
python run_tweet_summarizer.py username --device cuda
```

### Python API Usage

```python
from backend.tweet_summarizer_pipeline import TweetSummarizerPipeline

# Initialize pipeline
pipeline = TweetSummarizerPipeline()

# Process a user
results = pipeline.process_user("elonmusk", tweet_count=10)

# Print results
pipeline.print_results(results)

# Save results
pipeline.save_results(results)

# Clean up
pipeline.close()
```

### Jupyter Notebook

Open `tweet_summarizer_demo.ipynb` for an interactive demonstration with examples and analysis.

## 📊 System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Twitter       │    │   Fine-tuned     │    │   Reward        │
│   Scraper       │───▶│   Summarizer     │───▶│   Model         │
│   (Selenium)    │    │   (Qwen+LoRA)    │    │   (Quality)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Tweets    │    │   Generated      │    │   Quality       │
│   + Engagement  │    │   Summary        │    │   Score         │
│   Metrics       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔍 How It Works

1. **Tweet Scraping**: Uses Selenium WebDriver to navigate to Twitter profiles and scrape recent tweets
2. **Engagement Ranking**: Sorts tweets by total engagement (likes + retweets + replies)
3. **Text Preparation**: Combines top tweets into a structured format for summarization
4. **AI Summarization**: Passes the combined text through a fine-tuned Qwen model with LoRA weights
5. **Quality Assessment**: Scores the generated summary using a trained reward model
6. **Results Compilation**: Provides comprehensive output with statistics and metrics

## 📈 Output Format

The system provides detailed results including:

```json
{
  "username": "elonmusk",
  "tweets": [...],
  "combined_text": "Combined tweet content...",
  "summary": "Generated summary...",
  "score": 0.8542,
  "timestamp": "2024-01-15T10:30:00",
  "stats": {
    "tweet_count": 10,
    "original_length": 1250,
    "summary_length": 180,
    "compression_ratio": 0.144
  }
}
```

## 🎛️ Configuration Options

### Command Line Options

- `--count, -c`: Number of tweets to fetch (default: 10)
- `--max-length, -l`: Maximum summary length (default: 200)
- `--save, -s`: Save results to JSON file
- `--output, -o`: Specify output filename
- `--device, -d`: Choose device (auto/cuda/cpu/mps)
- `--temperature, -t`: Generation temperature (default: 0.7)
- `--quiet, -q`: Suppress detailed output

### Python API Options

```python
pipeline = TweetSummarizerPipeline(
    model_id="Qwen/Qwen1.5-0.5B",
    lora_weights_path="lora_weights.pt",
    reward_model_path="qwen_reward_model.pt",
    device="cuda"  # or "cpu", "mps", None for auto
)
```

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python test_pipeline.py
```

This will test:
- Model loading and initialization
- Tweet scraping functionality
- Summary generation
- Quality scoring
- Command-line interface

## 🔧 Troubleshooting

### Common Issues

1. **ChromeDriver not found**:
   - Install ChromeDriver and add to PATH
   - Or use `webdriver-manager`: `pip install webdriver-manager`

2. **Model files not found**:
   - Ensure `lora_weights.pt` and `qwen_reward_model.pt` are in the root directory
   - Check file permissions

3. **CUDA out of memory**:
   - Use `--device cpu` to run on CPU
   - Reduce batch size or max length

4. **Twitter scraping fails**:
   - The system falls back to sample data if scraping fails
   - Check your internet connection
   - Twitter may have rate limits or anti-bot measures

### Performance Tips

- Use GPU (`--device cuda`) for faster inference
- Reduce `--count` for faster processing
- Lower `--max-length` for shorter summaries
- Use `--quiet` mode for minimal output

## 📁 File Structure

```
RLHFSummarise/
├── backend/
│   ├── tweet_summarizer_pipeline.py  # Main pipeline class
│   ├── model.py                      # LoRA model setup
│   ├── reward.py                     # Reward model training/loading
│   ├── data_loader.py               # Data loading utilities
│   └── ...
├── twitter_scraper_selenium.py      # Twitter scraping functionality
├── run_tweet_summarizer.py         # Command-line interface
├── tweet_summarizer_demo.ipynb     # Jupyter notebook demo
├── test_pipeline.py                # Test suite
├── lora_weights.pt                 # Fine-tuned model weights
├── qwen_reward_model.pt            # Reward model weights
└── README.md                       # This file
```

## 🎯 Use Cases

- **Social Media Analysis**: Understand what a Twitter user is talking about
- **Content Curation**: Generate summaries of influential users' tweets
- **Research**: Analyze discourse patterns and engagement metrics
- **Personal Use**: Stay updated with favorite accounts efficiently
- **Brand Monitoring**: Track what's being said about topics or brands

## 🔮 Future Enhancements

- Support for other social media platforms
- Real-time monitoring and alerts
- Sentiment analysis integration
- Multi-language support
- Web interface
- Database storage for historical analysis
- Advanced filtering and ranking algorithms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the base language model
- **Hugging Face** for the Transformers library
- **LoRA** technique for efficient fine-tuning
- **Selenium** for web scraping capabilities

---

For more information, examples, and advanced usage, see the Jupyter notebook `tweet_summarizer_demo.ipynb`. 