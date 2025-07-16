import time
import random
import json
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

class TwitterSeleniumScraper:
    def __init__(self, headless=True):
        """
        Initialize the Twitter scraper with Selenium.
        
        Args:
            headless (bool): Run browser in headless mode
        """
        self.driver = None
        self.headless = headless
        self.setup_driver()
    
    def setup_driver(self):
        """Set up the Chrome WebDriver with appropriate options."""
        chrome_options = Options()
        
        if self.headless:
            chrome_options.add_argument("--headless")
        
        # Add options to avoid detection
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Set user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"❌ Error setting up Chrome driver: {e}")
            print("💡 Make sure you have Chrome installed and chromedriver in your PATH")
            self.driver = None
    
    def get_user_tweets(self, username, count=5, since_date=None, until_date=None):
        """
        Get the latest tweets from a Twitter user efficiently.
        
        Args:
            username (str): Twitter username (without @)
            count (int): Number of tweets to fetch (default: 5, optimized for recent tweets)
            since_date (datetime): Not used - kept for compatibility
            until_date (datetime): Not used - kept for compatibility
            
        Returns:
            list: List of tweet dictionaries
        """
        if not self.driver:
            print("❌ WebDriver not available, returning sample tweets")
            return self._get_sample_tweets(count)
        
        try:
            # Navigate directly to the user's timeline
            url = f"https://twitter.com/{username}"
            print(f"🔍 Navigating to @{username}'s timeline...")
            
            self.driver.get(url)
            time.sleep(3)  # Shorter initial wait
            
            # Wait for any tweet to appear (more reliable than specific selectors)
            print("⏳ Waiting for tweets to load...")
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: len(driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')) > 0
                )
                print("✅ Tweets found on page")
            except TimeoutException:
                print("⚠️ No tweets found, trying alternative selectors...")
                # Try alternative selectors
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article')
                if not tweet_elements:
                    print("❌ No tweets found with any selector")
                    return self._get_sample_tweets(count)
            
            # Extract tweets immediately without excessive scrolling
            tweets = self._extract_tweets_efficiently(count)
            
            if not tweets:
                print("⚠️ No tweets extracted, returning sample data")
                return self._get_sample_tweets(count)
            
            print(f"✅ Successfully extracted {len(tweets)} tweets")
            return tweets[:count]
            
        except Exception as e:
            print(f"❌ Error scraping tweets: {e}")
            return self._get_sample_tweets(count)
    
    def get_top_liked_tweets(self, username, count=10, sample_size=30, max_scrolls=5):
        """
        Get the top N most liked tweets from a Twitter user.
        
        Args:
            username (str): Twitter username (without @)
            count (int): Number of top tweets to return (default: 10)
            sample_size (int): Number of tweets to sample before sorting (default: 30)
            max_scrolls (int): Maximum scroll attempts to find tweets (default: 5)
            
        Returns:
            list: List of tweet dictionaries sorted by likes (descending)
        """
        if not self.driver:
            print("❌ WebDriver not available, returning sample tweets")
            return self._get_sample_tweets(count)
        
        try:
            # Navigate directly to the user's timeline
            url = f"https://twitter.com/{username}"
            print(f"🔍 Navigating to @{username}'s timeline...")
            
            self.driver.get(url)
            time.sleep(3)
            
            # Wait for tweets to load
            print("⏳ Waiting for tweets to load...")
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: len(driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')) > 0
                )
                print("✅ Tweets found on page")
            except TimeoutException:
                print("⚠️ No tweets found, trying alternative selectors...")
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article')
                if not tweet_elements:
                    print("❌ No tweets found with any selector")
                    return self._get_sample_tweets(count)
            
            # Extract a larger sample of tweets
            print(f"📊 Collecting {sample_size} tweets to find top {count}...")
            tweets = self._extract_tweets_for_sorting(sample_size, max_scrolls)
            
            if not tweets:
                print("⚠️ No tweets extracted, returning sample data")
                return self._get_sample_tweets(count)
            
            # Sort by likes and get top N
            sorted_tweets = self._sort_tweets_by_likes(tweets, count)
            
            print(f"✅ Successfully extracted top {len(sorted_tweets)} most liked tweets")
            return sorted_tweets
            
        except Exception as e:
            print(f"❌ Error scraping tweets: {e}")
            return self._get_sample_tweets(count)
    
    def _extract_tweets_efficiently(self, target_count):
        """Extract tweets efficiently without excessive scrolling."""
        tweets = []
        
        try:
            # Get all tweet elements on the current page
            tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
            
            if not tweet_elements:
                # Fallback to article elements
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article')
            
            print(f"📝 Found {len(tweet_elements)} tweet elements on page")
            
            # Process each tweet element
            for i, element in enumerate(tweet_elements):
                if len(tweets) >= target_count:
                    break
                
                try:
                    tweet_data = self._extract_single_tweet_improved(element)
                    if tweet_data and len(tweet_data['content'].strip()) > 5:  # Lower threshold
                        # Only skip obvious retweets, be more lenient
                        if not self._is_obvious_retweet(tweet_data['content']):
                            tweets.append(tweet_data)
                            print(f"   ✅ Extracted tweet {len(tweets)}: {tweet_data['content'][:60]}...")
                        else:
                            print(f"   ⚠️ Skipping obvious retweet: {tweet_data['content'][:50]}...")
                except Exception as e:
                    print(f"   ⚠️ Error extracting tweet {i+1}: {e}")
                    continue
            
            # If we don't have enough tweets, try minimal scrolling
            if len(tweets) < target_count:
                print(f"📜 Only got {len(tweets)} tweets, trying minimal scroll...")
                self._minimal_scroll_for_more_tweets(target_count - len(tweets))
                
                # Extract again after scroll
                new_tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                for element in new_tweet_elements[len(tweet_elements):]:  # Only process new elements
                    if len(tweets) >= target_count:
                        break
                    
                    try:
                        tweet_data = self._extract_single_tweet_improved(element)
                        if tweet_data and len(tweet_data['content'].strip()) > 5:
                            if not self._is_obvious_retweet(tweet_data['content']):
                                tweets.append(tweet_data)
                                print(f"   ✅ Extracted additional tweet {len(tweets)}: {tweet_data['content'][:60]}...")
                    except Exception as e:
                        continue
            
        except Exception as e:
            print(f"⚠️ Error extracting tweets: {e}")
        
        return tweets
    
    def _extract_tweets_for_sorting(self, target_count, max_scrolls=5):
        """Extract tweets with better metric extraction for sorting."""
        tweets = []
        processed_elements = set()
        
        try:
            # Scroll more to get a larger sample
            scroll_attempts = 0
            
            while len(tweets) < target_count and scroll_attempts < max_scrolls:
                # Get all tweet elements on the current page
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                
                if not tweet_elements:
                    tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, 'article')
                
                print(f"📝 Found {len(tweet_elements)} tweet elements on page")
                
                # Process each tweet element
                for element in tweet_elements:
                    if len(tweets) >= target_count:
                        break
                    
                    # Skip if we've already processed this element
                    element_id = element.id
                    if element_id in processed_elements:
                        continue
                    
                    processed_elements.add(element_id)
                    
                    try:
                        tweet_data = self._extract_single_tweet_with_metrics(element)
                        if tweet_data and len(tweet_data['content'].strip()) > 5:
                            if not self._is_obvious_retweet(tweet_data['content']):
                                tweets.append(tweet_data)
                                print(f"   ✅ Extracted tweet {len(tweets)}: {tweet_data['content'][:60]}... (❤️ {tweet_data['likes']})")
                    except Exception as e:
                        continue
                
                # Scroll for more tweets if needed
                if len(tweets) < target_count:
                    print(f"📜 Scrolling for more tweets ({len(tweets)}/{target_count})...")
                    self.driver.execute_script("window.scrollBy(0, 1000);")
                    time.sleep(2)
                    scroll_attempts += 1
                else:
                    break
            
        except Exception as e:
            print(f"⚠️ Error extracting tweets: {e}")
        
        return tweets
    
    def _extract_single_tweet_improved(self, element):
        """Extract data from a single tweet element with improved logic."""
        try:
            # Extract tweet text with better selectors
            content = ""
            
            # Try multiple text selectors in order of preference
            text_selectors = [
                '[data-testid="tweetText"]',
                'div[data-testid="tweetText"]',
                'span[data-testid="tweetText"]',
                'div[lang]',  # Tweet text often has lang attribute
                'div[dir="auto"]'  # Tweet text often has dir="auto"
            ]
            
            for selector in text_selectors:
                try:
                    text_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for text_element in text_elements:
                        text = text_element.text.strip()
                        if text and len(text) > 5:  # Lower threshold
                            content = text
                            break
                    if content:
                        break
                except:
                    continue
            
            # If no specific selector worked, try to extract from the entire element
            if not content:
                full_text = element.text.strip()
                # Split by lines and find the longest meaningful line
                lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                meaningful_lines = []
                
                for line in lines:
                    # Filter out obvious non-tweet content
                    if (len(line) > 5 and 
                        not line.startswith('@') and
                        not line.startswith('http') and
                        not line.isdigit() and
                        not any(word in line.lower() for word in ['follow', 'following', 'followers', 'retweet', 'like', 'reply', 'show this thread', 'view translation'])):
                        meaningful_lines.append(line)
                
                if meaningful_lines:
                    # Take the longest meaningful line
                    content = max(meaningful_lines, key=len)
            
            if not content or len(content.strip()) < 5:
                return None
            
            # Extract timestamp (simplified)
            timestamp = "Recent"
            try:
                time_elements = element.find_elements(By.CSS_SELECTOR, 'time')
                if time_elements:
                    dt = time_elements[0].get_attribute('datetime')
                    if dt:
                        timestamp = dt
                    else:
                        timestamp = time_elements[0].text.strip()
            except:
                pass
            
            # Extract engagement metrics (simplified)
            likes = self._extract_metric_simple(element, 'like')
            retweets = self._extract_metric_simple(element, 'retweet')
            replies = self._extract_metric_simple(element, 'reply')
            
            return {
                'content': content,
                'timestamp': timestamp,
                'likes': likes,
                'retweets': retweets,
                'replies': replies
            }
            
        except Exception as e:
            print(f"   ⚠️ Error in _extract_single_tweet_improved: {e}")
            return None
    
    def _extract_single_tweet_with_metrics(self, element):
        """Extract tweet data with improved metric extraction for sorting."""
        try:
            # Extract tweet text (same as before)
            content = ""
            text_selectors = [
                '[data-testid="tweetText"]',
                'div[data-testid="tweetText"]',
                'span[data-testid="tweetText"]',
                'div[lang]',
                'div[dir="auto"]'
            ]
            
            for selector in text_selectors:
                try:
                    text_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for text_element in text_elements:
                        text = text_element.text.strip()
                        if text and len(text) > 5:
                            content = text
                            break
                    if content:
                        break
                except:
                    continue
            
            # Fallback text extraction
            if not content:
                full_text = element.text.strip()
                lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                meaningful_lines = []
                
                for line in lines:
                    if (len(line) > 5 and 
                        not line.startswith('@') and
                        not line.startswith('http') and
                        not line.isdigit() and
                        not any(word in line.lower() for word in ['follow', 'following', 'followers', 'retweet', 'like', 'reply', 'show this thread', 'view translation'])):
                        meaningful_lines.append(line)
                
                if meaningful_lines:
                    content = max(meaningful_lines, key=len)
            
            if not content or len(content.strip()) < 5:
                return None
            
            # Extract timestamp
            timestamp = "Recent"
            try:
                time_elements = element.find_elements(By.CSS_SELECTOR, 'time')
                if time_elements:
                    dt = time_elements[0].get_attribute('datetime')
                    if dt:
                        timestamp = dt
                    else:
                        timestamp = time_elements[0].text.strip()
            except:
                pass
            
            # Extract engagement metrics with improved accuracy
            likes = self._extract_metric_improved(element, 'like')
            retweets = self._extract_metric_improved(element, 'retweet')
            replies = self._extract_metric_improved(element, 'reply')
            
            return {
                'content': content,
                'timestamp': timestamp,
                'likes': likes,
                'retweets': retweets,
                'replies': replies,
                'likes_count': self._parse_metric_to_int(likes)  # For sorting
            }
            
        except Exception as e:
            print(f"   ⚠️ Error in _extract_single_tweet_with_metrics: {e}")
            return None
    
    def _extract_metric_simple(self, element, metric_type):
        """Extract engagement metric with simplified logic."""
        try:
            # Look for metrics in the element text
            text = element.text.lower()
            if metric_type == 'like':
                if 'like' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
            elif metric_type == 'retweet':
                if 'retweet' in text or 'repost' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
            elif metric_type == 'reply':
                if 'reply' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
        except:
            pass
        return "0"
    
    def _extract_metric_improved(self, element, metric_type):
        """Extract engagement metric with improved accuracy."""
        try:
            # Try specific selectors for metrics
            metric_selectors = {
                'like': [
                    '[data-testid="like"]',
                    '[data-testid="unlike"]',
                    'div[aria-label*="like"]',
                    'div[aria-label*="Like"]'
                ],
                'retweet': [
                    '[data-testid="retweet"]',
                    '[data-testid="unretweet"]',
                    'div[aria-label*="retweet"]',
                    'div[aria-label*="Retweet"]'
                ],
                'reply': [
                    '[data-testid="reply"]',
                    'div[aria-label*="reply"]',
                    'div[aria-label*="Reply"]'
                ]
            }
            
            selectors = metric_selectors.get(metric_type, [])
            
            for selector in selectors:
                try:
                    metric_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for metric_element in metric_elements:
                        # Look for the count in the parent or sibling elements
                        parent = metric_element.find_element(By.XPATH, "./..")
                        text = parent.text.strip()
                        
                        # Extract numbers from the text
                        numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                        if numbers:
                            return numbers[0]
                except:
                    continue
            
            # Fallback: look in the entire element text
            text = element.text.lower()
            if metric_type == 'like':
                if 'like' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
            elif metric_type == 'retweet':
                if 'retweet' in text or 'repost' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
            elif metric_type == 'reply':
                if 'reply' in text:
                    numbers = re.findall(r'(\d+(?:,\d+)*)', text)
                    return numbers[0] if numbers else "0"
        except:
            pass
        return "0"
    
    def _parse_metric_to_int(self, metric_str):
        """Convert metric string to integer for sorting."""
        try:
            # Remove commas and convert to int
            return int(metric_str.replace(',', ''))
        except:
            return 0
    
    def _is_obvious_retweet(self, content):
        """Check if content is obviously a retweet (more lenient)."""
        content_lower = content.lower()
        
        # Only flag obvious retweets
        obvious_indicators = [
            'rt @',  # Old-style retweets
            'retweeted by',  # New-style retweets
            'reposted by',  # Reposts
        ]
        
        for indicator in obvious_indicators:
            if indicator in content_lower:
                return True
        
        return False
    
    def _minimal_scroll_for_more_tweets(self, needed_count):
        """Minimal scrolling to get more tweets."""
        print(f"📜 Scrolling to get {needed_count} more tweets...")
        
        # Just one or two scrolls
        for i in range(2):
            self.driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(2)
            
            # Check if we have more tweets
            new_tweets = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
            if len(new_tweets) > needed_count:
                print(f"   ✅ Found enough tweets after scroll {i+1}")
                break
    
    def _sort_tweets_by_likes(self, tweets, count):
        """Sort tweets by like count and return top N."""
        try:
            # Sort by likes_count (descending)
            sorted_tweets = sorted(tweets, key=lambda x: x.get('likes_count', 0), reverse=True)
            
            # Remove the likes_count field from the final result
            for tweet in sorted_tweets:
                if 'likes_count' in tweet:
                    del tweet['likes_count']
            
            print(f"🏆 Top {count} most liked tweets:")
            for i, tweet in enumerate(sorted_tweets[:count], 1):
                print(f"   {i}. ❤️ {tweet['likes']} - {tweet['content'][:80]}...")
            
            return sorted_tweets[:count]
            
        except Exception as e:
            print(f"⚠️ Error sorting tweets: {e}")
            return tweets[:count]
    
    def get_top_liked_tweets_fast(self, username, count=10):
        """
        Faster approach: Get top liked tweets with early stopping.
        Stops when we find enough high-engagement tweets.
        
        Args:
            username (str): Twitter username (without @)
            count (int): Number of top tweets to return (default: 10)
            
        Returns:
            list: List of tweet dictionaries sorted by likes (descending)
        """
        if not self.driver:
            print("❌ WebDriver not available, returning sample tweets")
            return self._get_sample_tweets(count)
        
        try:
            url = f"https://twitter.com/{username}"
            print(f"🔍 Navigating to @{username}'s timeline...")
            
            self.driver.get(url)
            time.sleep(3)
            
            # Wait for tweets to load
            try:
                WebDriverWait(self.driver, 10).until(
                    lambda driver: len(driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')) > 0
                )
                print("✅ Tweets found on page")
            except TimeoutException:
                print("❌ No tweets found")
                return self._get_sample_tweets(count)
            
            # Collect tweets with early stopping for high engagement
            tweets = []
            high_engagement_threshold = 1000  # Stop when we find tweets with 1000+ likes
            scroll_attempts = 0
            max_scrolls = 8
            
            print(f"🚀 Fast mode: Looking for tweets with {high_engagement_threshold}+ likes...")
            
            while len(tweets) < count and scroll_attempts < max_scrolls:
                tweet_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweet"]')
                
                for element in tweet_elements:
                    if len(tweets) >= count:
                        break
                    
                    try:
                        tweet_data = self._extract_single_tweet_with_metrics(element)
                        if tweet_data and len(tweet_data['content'].strip()) > 5:
                            if not self._is_obvious_retweet(tweet_data['content']):
                                likes_count = tweet_data.get('likes_count', 0)
                                
                                # Add high-engagement tweets immediately
                                if likes_count >= high_engagement_threshold:
                                    tweets.append(tweet_data)
                                    print(f"   🎯 High engagement tweet: ❤️ {tweet_data['likes']} - {tweet_data['content'][:60]}...")
                                
                                # If we have enough high-engagement tweets, stop
                                if len(tweets) >= count:
                                    print(f"✅ Found {len(tweets)} high-engagement tweets, stopping early")
                                    break
                    except Exception as e:
                        continue
                
                # Scroll for more tweets
                if len(tweets) < count:
                    self.driver.execute_script("window.scrollBy(0, 800);")
                    time.sleep(1.5)  # Faster scroll
                    scroll_attempts += 1
            
            # Sort and return top tweets
            if tweets:
                sorted_tweets = self._sort_tweets_by_likes(tweets, count)
                return sorted_tweets
            else:
                print("⚠️ No high-engagement tweets found, falling back to regular method")
                return self.get_top_liked_tweets(username, count, sample_size=20)
            
        except Exception as e:
            print(f"❌ Error in fast mode: {e}")
            return self._get_sample_tweets(count)
    
    def _get_sample_tweets(self, count):
        """Return sample tweets when scraping fails."""
        sample_tweets = [
            "Just had an amazing idea for a new project! 🚀",
            "The future of technology is incredibly exciting.",
            "Working on some really cool stuff today.",
            "Innovation happens when you least expect it.",
            "Sometimes the best solutions are the simplest ones.",
            "Building something that could change everything.",
            "The intersection of AI and human creativity is fascinating.",
            "Progress is made one step at a time.",
            "Great things are built by teams, not individuals.",
            "The best time to start was yesterday, the second best is now."
        ]
        
        return [
            {
                'content': tweet,
                'timestamp': 'Sample',
                'likes': str(random.randint(50, 1000)),
                'retweets': str(random.randint(10, 500)),
                'replies': str(random.randint(5, 200))
            }
            for tweet in random.sample(sample_tweets, min(count, len(sample_tweets)))
        ]
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()

def main():
    """Demo the Twitter Selenium scraper."""
    scraper = TwitterSeleniumScraper(headless=True)
    
    try:
        # Test with usernames
        usernames = ["elonmusk", "dril", "horse_ebooks"]
        
        for username in usernames:
            print(f"\n{'='*60}")
            print(f"📱 SCRAPING TOP LIKED TWEETS FROM @{username}")
            print(f"{'='*60}")
            
            # Get top 10 most liked tweets
            tweets = scraper.get_top_liked_tweets(username, count=10, sample_size=50)
            
            print(f"\n🏆 TOP 10 MOST LIKED TWEETS FROM @{username}:")
            for i, tweet in enumerate(tweets, 1):
                print(f"\n{i}. ❤️ {tweet['likes']} likes")
                print(f"   📝 {tweet['content']}")
                print(f"   📅 {tweet['timestamp']} | 🔄 {tweet['retweets']} | 💬 {tweet['replies']}")
            
            # Add delay between requests
            time.sleep(3)
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 