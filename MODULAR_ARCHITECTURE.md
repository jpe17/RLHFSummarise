# Modular Pipeline Architecture

## Overview

This document describes the refactored modular architecture for the RLHF Summarization Pipeline. The new architecture provides a clean, extensible framework that supports multiple social media platforms with minimal code changes.

## 🏗️ Architecture Overview

The modular architecture consists of the following core components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Application                          │
│                     (Flask + SocketIO)                         │
└─────────────────────────┬───────────────────────────────────────┘
                         │
┌─────────────────────────▼───────────────────────────────────────┐
│                   Pipeline Manager                             │
│                   (Orchestrator)                               │
└─────────────────────────┬───────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Crawlers  │  │   Storage   │  │ Processors  │
│             │  │             │  │             │
│ • Twitter   │  │ • JSON      │  │ • Content   │
│ • Instagram │  │ • Database  │  │ • Image2Text│
│ • TikTok    │  │ • Cloud     │  │ • Selector  │
└─────────────┘  └─────────────┘  └─────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Summarizer  │  │ Voice Synth │  │ Data Models │
│             │  │             │  │             │
│ • RLHF/PPO  │  │ • TTS       │  │ • SocialPost│
│ • Simple    │  │ • Mock      │  │ • Summary   │
│ • Custom    │  │ • Cloud     │  │ • VoiceOut  │
└─────────────┘  └─────────────┘  └─────────────┘
```

## 📁 Project Structure

```
RLHFSummarise/
├── core/                          # Core framework
│   ├── __init__.py
│   ├── data_models.py             # Common data models
│   ├── interfaces.py              # Abstract base classes
│   └── pipeline_manager.py        # Main orchestrator
├── implementations/               # Component implementations
│   ├── __init__.py
│   ├── storage.py                 # Storage implementations
│   ├── crawlers.py                # Platform crawlers
│   ├── processors.py              # Content processors
│   ├── summarizer.py              # Summarization components
│   └── voice_synthesizer.py       # Voice synthesis components
├── pipeline_factory.py            # Factory for creating pipelines
├── modular_app.py                 # New Flask application
├── example_usage.py               # Usage examples
└── MODULAR_ARCHITECTURE.md        # This documentation
```

## 🔧 Core Components

### 1. Data Models (`core/data_models.py`)

**Universal Data Format**: All platforms use the same data structure:

```python
@dataclass
class SocialPost:
    id: str
    platform: Platform              # TWITTER, INSTAGRAM, TIKTOK
    username: str
    content: str
    content_type: ContentType        # TEXT, IMAGE, VIDEO, MIXED
    media_items: List[MediaItem]     # Images, videos, etc.
    timestamp: datetime
    likes: int
    shares: int
    comments: int
    platform_data: Dict[str, Any]   # Platform-specific metadata
```

**Processing Pipeline**:
- `ProcessedContent` → `Summary` → `VoiceOutput` → `PipelineResult`

### 2. Interfaces (`core/interfaces.py`)

Abstract base classes ensure consistent behavior:

- `BaseCrawler` - Platform-specific crawling
- `BaseStorage` - Data persistence
- `BaseContentProcessor` - Content preparation
- `BaseImageToText` - Image captioning
- `BaseSummarizer` - Text summarization
- `BaseVoiceSynthesizer` - Voice synthesis
- `BasePostSelector` - Post selection strategies

### 3. Pipeline Manager (`core/pipeline_manager.py`)

Central orchestrator that:
- Manages component registration
- Coordinates the processing flow
- Handles progress tracking
- Provides unified API

## 🚀 Getting Started

### 1. Basic Usage

```python
from pipeline_factory import PipelineFactory
from core.data_models import Platform

# Create a pipeline
pipeline = PipelineFactory.create_production_pipeline()

# Process posts
result = pipeline.run_full_pipeline(
    platform=Platform.TWITTER,
    username="elonmusk",
    voice_name="freeman",
    selection_type="top",
    count=5
)

print(f"Summary: {result.summary.content}")
print(f"Audio: {result.voice_output.audio_path}")
```

### 2. Custom Configuration

```python
config = {
    "data_dir": "custom_data",
    "summarizer_type": "rlhf",
    "voice_synthesizer_type": "tts",
    "use_vision_model": True,
    "crawlers": {
        "twitter": {"headless": True},
        "instagram": {"api_key": "your_key"}
    }
}

pipeline = PipelineFactory.create_custom_pipeline(config)
```

### 3. Web Application

```bash
# Run the new modular web app
python modular_app.py

# Access at http://localhost:5464
```

## 🔌 Adding New Platforms

### 1. Create Platform Crawler

```python
class TikTokCrawler(BaseCrawler):
    @property
    def platform(self) -> Platform:
        return Platform.TIKTOK
        
    def crawl_user(self, username: str, count: int = 10) -> List[SocialPost]:
        # Implementation here
        pass
        
    def validate_username(self, username: str) -> bool:
        # Implementation here
        pass
```

### 2. Register with Factory

```python
# In pipeline_factory.py
crawler = TikTokCrawler()
manager.register_crawler(crawler)
```

### 3. Handle Platform-Specific Features

```python
# Platform-specific processing in ContentProcessor
if post.platform == Platform.TIKTOK:
    # Handle TikTok-specific features
    pass
```

## 🎨 Customizing Components

### 1. Custom Summarizer

```python
class CustomSummarizer(BaseSummarizer):
    def generate_summary(self, content: ProcessedContent, 
                        max_length: int = 200) -> Summary:
        # Your custom summarization logic
        pass
```

### 2. Custom Voice Synthesizer

```python
class CustomVoiceSynthesizer(BaseVoiceSynthesizer):
    def synthesize(self, text: str, voice_name: str) -> VoiceOutput:
        # Your custom voice synthesis logic
        pass
```

### 3. Custom Storage

```python
class DatabaseStorage(BaseStorage):
    def store_posts(self, posts: List[SocialPost]) -> bool:
        # Store in database instead of JSON
        pass
```

## 🧪 Testing

### Test Pipeline

```python
# Create lightweight pipeline for testing
pipeline = PipelineFactory.create_testing_pipeline()

# Uses mock components - no heavy dependencies
```

### Run Examples

```bash
python example_usage.py
```

## 📊 Data Migration

### Migrate Existing Twitter Data

```python
from implementations.storage import MigrationHelper

# Migrate old Twitter JSON format to new universal format
helper = MigrationHelper()
helper.migrate_twitter_data(storage)
```

## 🔍 Debugging & Monitoring

### 1. Progress Tracking

```python
def progress_callback(progress: int, message: str):
    print(f"Progress: {progress}% - {message}")

pipeline.set_progress_callback(progress_callback)
```

### 2. Component Validation

```python
validation = pipeline.validate_configuration()
print(f"Pipeline status: {validation}")
```

### 3. Configuration Validation

```python
config_validation = PipelineFactory.validate_config(config)
if not config_validation["valid"]:
    print(f"Errors: {config_validation['errors']}")
```

## 🚦 API Endpoints

### New Modular Web API

- `POST /api/initialize` - Initialize pipeline
- `GET /api/platforms` - Get available platforms
- `GET /api/users/<platform>` - Get users for platform
- `POST /api/preview-posts` - Preview posts with selection
- `POST /api/process` - Run full pipeline
- `POST /api/crawl` - Crawl new posts
- `GET /api/stats/<platform>/<username>` - Get user statistics

## 🎯 Benefits of New Architecture

### 1. **Modularity**
- Easy to swap components
- Clear separation of concerns
- Testable components

### 2. **Extensibility**
- Add new platforms easily
- Custom processing pipelines
- Pluggable components

### 3. **Maintainability**
- Clear interfaces
- Comprehensive documentation
- Consistent error handling

### 4. **Platform Agnostic**
- Universal data format
- Reusable summarization
- Consistent voice synthesis

### 5. **Instagram Support**
- Image-to-text processing
- Mixed content handling
- Media metadata extraction

## 📈 Performance Considerations

### 1. **Lazy Loading**
- Components initialized on demand
- Reduced memory footprint
- Faster startup times

### 2. **Parallel Processing**
- Background processing
- Progress tracking
- Non-blocking operations

### 3. **Caching**
- Model caching
- Result caching
- Configuration caching

## 🔮 Future Enhancements

### 1. **Additional Platforms**
- TikTok implementation
- LinkedIn support
- YouTube integration

### 2. **Advanced Features**
- Real-time processing
- Batch processing
- Scheduled crawling

### 3. **Deployment**
- Docker containers
- Cloud deployment
- Microservices architecture

## 🤝 Contributing

### 1. **Adding New Platforms**
1. Implement `BaseCrawler` interface
2. Handle platform-specific data
3. Update factory registration
4. Add tests

### 2. **Improving Components**
1. Implement relevant base class
2. Add configuration options
3. Update factory
4. Document changes

### 3. **Code Standards**
- Follow existing patterns
- Add type hints
- Include docstrings
- Write tests

## 📚 References

- `example_usage.py` - Complete usage examples
- `core/interfaces.py` - Interface documentation
- `pipeline_factory.py` - Configuration options
- `modular_app.py` - Web API implementation

---

**Note**: This modular architecture maintains backward compatibility while providing a clean path for future enhancements. The old `app.py` continues to work, while the new `modular_app.py` provides the enhanced functionality. 