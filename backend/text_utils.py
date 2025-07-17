import re
import emoji

def clean_text_for_tts(text):
    """
    Clean text for better TTS synthesis by removing:
    - Hashtags (#hashtag)
    - Emojis and emoticons
    - URLs and Twitter links
    - Excessive punctuation
    - Twitter handles (@username)
    - RT (retweet indicators)
    - Twitter-specific formatting
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text suitable for TTS
    """
    if not text:
        return ""
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove hashtags completely (both # and content) - more robust pattern
    text = re.sub(r'#\w+', '', text)
    
    # Remove @ symbol but keep the username text
    text = re.sub(r'@(\w+)', r'\1', text)
    
    # Remove URLs and Twitter links more comprehensively
    # Match various URL patterns including Twitter's t.co links
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'pic\.twitter\.com/[^\s]+', '', text)
    text = re.sub(r't\.co/[^\s]+', '', text)
    
    # Remove RT indicators
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    
    # Remove Twitter-specific patterns like "— @username (username) date"
    text = re.sub(r'—\s*@\w+\s*\(\w+\)\s*\w+\s+\d+', '', text)
    text = re.sub(r'—\s*@\w+', '', text)
    
    # Remove date patterns at the end
    text = re.sub(r'\w+\s+\d+,\s+\d{4}$', '', text)
    text = re.sub(r'\d{4}$', '', text)
    
    # Remove patterns like "(@username) date" or "(username) date"
    text = re.sub(r'\(\s*@?\w+\s*\)\s*\w+\s+\d+', '', text)
    text = re.sub(r'\(\s*@?\w+\s*\)', '', text)
    
    # Remove emoticons like :), :(, :D, etc.
    text = re.sub(r':[)\-DdPpOo]', '', text)
    text = re.sub(r'[)\-DdPpOo]:', '', text)
    
    # Remove other common emoticons
    emoticon_patterns = [
        r'[;:]-?[)DdPpOo]',  # ;), ;D, etc.
        r'[)DdPpOo]-?[;:]',  # ):, D:, etc.
        r'[<>][;:]-?[)DdPpOo]',  # <3, >:), etc.
        r'[;:]-?[<>]',  # ;<, :>, etc.
    ]
    for pattern in emoticon_patterns:
        text = re.sub(pattern, '', text)
    
    # Clean up excessive punctuation
    # Replace multiple periods/commas with single ones
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    # Remove leading/trailing punctuation
    text = re.sub(r'^[.,!?;:\s]+', '', text)
    text = re.sub(r'[.,!?;:\s]+$', '', text)
    
    # Remove empty parentheses and brackets
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    
    # Ensure proper sentence endings
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    # Final cleanup of any remaining artifacts
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    
    return text

def clean_tweet_content(tweet_content):
    """
    Clean individual tweet content for better summarization.
    This removes noise while keeping essential content.
    
    Args:
        tweet_content (str): Raw tweet content
        
    Returns:
        str: Cleaned tweet content
    """
    if not tweet_content:
        return ""
    
    # Remove emojis
    text = emoji.replace_emoji(tweet_content, replace='')
    
    # Remove hashtags completely
    text = re.sub(r'#\w+', '', text)
    
    # Remove @ symbol but keep the username text
    text = re.sub(r'@(\w+)', r'\1', text)
    
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'pic\.twitter\.com/[^\s]+', '', text)
    text = re.sub(r't\.co/[^\s]+', '', text)
    
    # Remove RT indicators
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    
    # Remove emoticons
    text = re.sub(r':[)\-DdPpOo]', '', text)
    text = re.sub(r'[)\-DdPpOo]:', '', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_text_for_summarization(text):
    """
    Clean text specifically for summarization tasks.
    This removes noise while preserving important content.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text suitable for summarization
    """
    if not text:
        return ""
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # More aggressive hashtag removal - multiple patterns
    text = re.sub(r'#\w+', '', text)  # Standard hashtags
    text = re.sub(r'#[^\s]+', '', text)  # Hashtags with special chars
    text = re.sub(r'\s#\w+', ' ', text)  # Hashtags with leading space
    text = re.sub(r'\s#[^\s]+', ' ', text)  # Any hashtag pattern
    
    # Remove @ mentions more thoroughly
    text = re.sub(r'@\w+', '', text)  # Remove @ mentions completely
    text = re.sub(r'@[^\s]+', '', text)  # Remove @ mentions with special chars
    
    # Remove URLs more comprehensively
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'pic\.twitter\.com/[^\s]+', '', text)
    text = re.sub(r't\.co/[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    
    # Remove RT indicators and quote tweet patterns
    text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'RT @\w+:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'QT @\w+:', '', text, flags=re.IGNORECASE)
    
    # Remove emoticons like :), :(, :D, etc.
    text = re.sub(r':[)\-DdPpOo]', '', text)
    text = re.sub(r'[)\-DdPpOo]:', '', text)
    
    # Remove common Twitter artifacts
    text = re.sub(r'\.\.\.$', '', text)  # Trailing dots
    text = re.sub(r'^\.\.\.$', '', text)  # Leading dots
    text = re.sub(r'\s+\.\.\.$', '', text)  # Spaced trailing dots
    
    # Remove excessive punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    # Remove lines that are just punctuation or very short
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 5 and not re.match(r'^[^\w]*$', line):  # Not just punctuation
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    return text

def format_tweet_for_summarization(tweets):
    """
    Format a list of tweets into a clean text for summarization.
    
    Args:
        tweets (list): List of tweet dictionaries
        
    Returns:
        str: Formatted and cleaned text for summarization
    """
    if not tweets:
        return ""
    
    formatted_text = ""
    for i, tweet in enumerate(tweets, 1):
        content = tweet.get('content', '')
        if content:
            # Clean the tweet content
            cleaned_content = clean_tweet_content(content)
            if cleaned_content:
                formatted_text += f"{cleaned_content}\n\n"
    
    # Final cleaning for the entire text
    return clean_text_for_summarization(formatted_text) 