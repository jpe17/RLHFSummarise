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
    
    def get_user_tweets(self, username, count=10, since_date=None, until_date=None):
        """
        Get the latest tweets from a Twitter user.
        
        Args:
            username (str): Twitter username (without @)
            count (int): Number of tweets to fetch (default: 10)
            since_date (datetime): Not used - kept for compatibility
            until_date (datetime): Not used - kept for compatibility
            
        Returns:
            list: List of tweet dictionaries
        """
        if not self.driver:
            print("❌ WebDriver not available, returning sample tweets")
            return self._get_sample_tweets(count)
        
        try:
            # Navigate to the user's profile - specifically to their posts tab to get original tweets only
            url = f"https://twitter.com/{username}"
            print(f"🔍 Navigating to @{username}'s profile...")
            
            self.driver.get(url)
            time.sleep(5)  # Longer initial wait
            
            # Click on "Posts" tab to get only original tweets (not retweets/replies)
            try:
                print("📌 Looking for Posts tab to get original tweets only...")
                posts_tab_selectors = [
                    'a[href$="/posts"]',  # Direct posts tab link
                    'a[href*="/' + username + '"][href$="/posts"]',  # More specific posts tab
                    'nav a[role="tab"]:first-child',  # First tab is usually posts
                    'div[role="tablist"] a:first-child'  # Alternative tablist structure
                ]
                
                posts_tab_clicked = False
                for selector in posts_tab_selectors:
                    try:
                        posts_tab = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        posts_tab.click()
                        print(f"   ✅ Clicked Posts tab using selector: {selector}")
                        posts_tab_clicked = True
                        time.sleep(3)  # Wait for posts to load
                        break
                    except:
                        continue
                
                if not posts_tab_clicked:
                    print("   ⚠️ Could not find Posts tab, will get mixed content")
                    
            except Exception as e:
                print(f"   ⚠️ Error clicking Posts tab: {e}")
                print("   ⚠️ Proceeding with default timeline (may include retweets)")
            
            # Wait for tweets to load with multiple selectors
            print("⏳ Waiting for tweets to load...")
            tweet_loaded = False
            selectors_to_try = [
                '[data-testid="tweet"]',
                'article[data-testid="tweet"]',
                '[role="article"]',
                'article'
            ]
            
            for selector in selectors_to_try:
                try:
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    print(f"   ✅ Found tweets using selector: {selector}")
                    tweet_loaded = True
                    break
                except TimeoutException:
                    print(f"   ⚠️ No tweets found with selector: {selector}")
                    continue
            
            if not tweet_loaded:
                print("   ⚠️ No tweets found on page")
                return self._get_sample_tweets(count)
            
            # Scroll to load more tweets
            self._scroll_to_load_tweets(count)
            
            # Extract tweets
            tweets = self._extract_tweets_from_page(count)
            
            if not tweets:
                print("⚠️ No tweets extracted, returning sample data")
                return self._get_sample_tweets(count)
            
            print(f"✅ Successfully extracted {len(tweets)} tweets")
            return tweets[:count]
            
        except Exception as e:
            print(f"❌ Error scraping tweets: {e}")
            return self._get_sample_tweets(count)
    
    def _scroll_to_load_tweets(self, target_count):
        """Scroll down to load more tweets."""
        print("📜 Scrolling to load tweets...")
        
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        for i in range(10):  # Increased scroll attempts
            # Scroll to bottom
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)  # Longer wait for content to load
            
            # Check how many tweets we have with multiple selectors
            tweet_count = 0
            selectors = [
                '[data-testid="tweet"]',
                'article[data-testid="tweet"]', 
                '[role="article"]',
                'article'
            ]
            
            for selector in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    tweet_count = len(elements)
                    break
            
            print(f"   Found {tweet_count} tweet elements after scroll {i+1}")
            
            if tweet_count >= target_count:
                print(f"   ✅ Found enough tweets ({tweet_count} >= {target_count})")
                break
            
            # Check if page height changed (new content loaded)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                print(f"   ⚠️ No new content loaded, trying alternative scroll...")
                # Try alternative scrolling methods
                self.driver.execute_script("window.scrollBy(0, 1000);")
                time.sleep(2)
                # Try pressing End key
                from selenium.webdriver.common.keys import Keys
                from selenium.webdriver.common.action_chains import ActionChains
                ActionChains(self.driver).send_keys(Keys.END).perform()
                time.sleep(2)
            
            last_height = new_height
    
    def _extract_tweets_from_page(self, count):
        """Extract tweets from the current page."""
        tweets = []
        
        try:
            # Try multiple selectors to find tweet elements
            selectors = [
                '[data-testid="tweet"]',
                'article[data-testid="tweet"]',
                '[role="article"]',
                'article'
            ]
            
            tweet_elements = []
            for selector in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    tweet_elements = elements
                    print(f"   Found {len(elements)} tweet elements using selector: {selector}")
                    break
            
            if not tweet_elements:
                print("   ⚠️ No tweet elements found with any selector")
                return tweets
            
            # Extract data from each tweet element
            for i, element in enumerate(tweet_elements):
                if len(tweets) >= count:
                    break
                    
                try:
                    tweet_data = self._extract_single_tweet(element)
                    if tweet_data and len(tweet_data['content'].strip()) > 10:
                        # Check if this is a retweet/repost (try to filter them out)
                        is_retweet = self._is_retweet(element)
                        if is_retweet:
                            print(f"   ⚠️ Skipping retweet: {tweet_data['content'][:50]}...")
                            continue
                        
                        # Avoid duplicates
                        if not any(t['content'] == tweet_data['content'] for t in tweets):
                            tweets.append(tweet_data)
                            print(f"   ✅ Extracted original tweet {len(tweets)}: {tweet_data['content'][:50]}...")
                except Exception as e:
                    print(f"   ⚠️ Error extracting tweet {i+1}: {e}")
                    continue
            
        except Exception as e:
            print(f"⚠️ Error extracting tweets: {e}")
        
        return tweets
    
    def _extract_single_tweet(self, element):
        """Extract data from a single tweet element."""
        try:
            # Extract tweet text with multiple approaches
            content = ""
            
            # Try primary tweet text selector
            text_selectors = [
                '[data-testid="tweetText"]',
                'div[data-testid="tweetText"]',
                'span[data-testid="tweetText"]',
                '[lang]',  # Often tweet text has lang attribute
                'div[dir="auto"]'  # Tweet text often has dir="auto"
            ]
            
            for selector in text_selectors:
                try:
                    text_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for text_element in text_elements:
                        text = text_element.text.strip()
                        if text and len(text) > 10:
                            content = text
                            break
                    if content:
                        break
                except:
                    continue
            
            # Fallback to element text if no specific selector worked
            if not content:
                content = element.text.strip()
                # Clean up the content
                lines = content.split('\n')
                filtered_lines = []
                for line in lines:
                    line = line.strip()
                    if (line and len(line) > 10 and 
                        not line.startswith('@') and
                        not line.startswith('#') and
                        not line.startswith('http') and
                        not line.isdigit() and
                        not any(word in line.lower() for word in ['follow', 'following', 'followers', 'retweet', 'like', 'reply', 'show this thread'])):
                        filtered_lines.append(line)
                
                if filtered_lines:
                    content = filtered_lines[0]  # Take the first meaningful line
            
            if not content or len(content.strip()) < 10:
                return None
            
            # Extract timestamp with multiple approaches
            timestamp = "Unknown"
            timestamp_selectors = [
                'time',
                '[datetime]',
                'a[href*="/status/"]',  # Tweet links often contain timestamp
                'span[title]'  # Sometimes timestamp is in title attribute
            ]
            
            for selector in timestamp_selectors:
                try:
                    time_elements = element.find_elements(By.CSS_SELECTOR, selector)
                    for time_element in time_elements:
                        # Try datetime attribute first
                        dt = time_element.get_attribute('datetime')
                        if dt:
                            timestamp = dt
                            break
                        # Try title attribute
                        title = time_element.get_attribute('title')
                        if title and any(word in title.lower() for word in ['am', 'pm', '202']):
                            timestamp = title
                            break
                        # Try text content
                        text = time_element.text.strip()
                        if text and any(word in text.lower() for word in ['ago', 'min', 'hour', 'day', 'week', 'month', 'year']):
                            timestamp = text
                            break
                    if timestamp != "Unknown":
                        break
                except:
                    continue
            
            # Extract engagement metrics
            likes = self._extract_metric(element, '[data-testid="like"]')
            retweets = self._extract_metric(element, '[data-testid="retweet"]')
            replies = self._extract_metric(element, '[data-testid="reply"]')
            
            return {
                'content': content,
                'timestamp': timestamp,
                'likes': likes,
                'retweets': retweets,
                'replies': replies
            }
            
        except Exception as e:
            print(f"   ⚠️ Error in _extract_single_tweet: {e}")
            return None
    
    def _extract_metric(self, element, selector):
        """Extract engagement metric from tweet element."""
        try:
            metric_element = element.find_element(By.CSS_SELECTOR, selector)
            text = metric_element.text.strip()
            numbers = re.findall(r'[\d,]+', text)
            return numbers[0] if numbers else "0"
        except:
            return "0"
    
    def _is_retweet(self, element):
        """Check if this tweet element is a retweet/repost."""
        try:
            # Look for retweet indicators
            retweet_indicators = [
                'svg[data-testid="retweet"]',  # Retweet icon
                '[data-testid="socialContext"]',  # "Username retweeted" text
                'span:contains("retweeted")',  # Text containing "retweeted"
                'span:contains("Retweeted")',  # Text containing "Retweeted"
                '[aria-label*="retweet"]',  # Aria labels mentioning retweet
                '[aria-label*="Retweet"]'   # Aria labels mentioning Retweet
            ]
            
            for indicator in retweet_indicators:
                try:
                    if indicator.startswith('span:contains'):
                        # Handle text-based detection
                        spans = element.find_elements(By.TAG_NAME, 'span')
                        for span in spans:
                            if 'retweet' in span.text.lower():
                                return True
                    else:
                        # Handle CSS selector detection
                        if element.find_elements(By.CSS_SELECTOR, indicator):
                            return True
                except:
                    continue
            
            # Also check if the tweet text starts with "RT @" (old-style retweets)
            try:
                text_element = element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                if text_element.text.strip().startswith('RT @'):
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            # If we can't determine, assume it's not a retweet
            return False
    
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
            print(f"📱 SCRAPING TWEETS FROM @{username}")
            print(f"{'='*60}")
            
            tweets = scraper.get_user_tweets(username, count=10)
            
            for i, tweet in enumerate(tweets, 1):
                print(f"\n{i}. {tweet['content']}")
                print(f"   📅 {tweet['timestamp']} | ❤️ {tweet['likes']} | 🔄 {tweet['retweets']} | 💬 {tweet['replies']}")
            
            # Add delay between requests
            time.sleep(3)
    
    finally:
        scraper.close()

if __name__ == "__main__":
    main() 