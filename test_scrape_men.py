import requests
import time
from bs4 import BeautifulSoup
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HMScraper:
    def __init__(self):
        self.base_url = 'https://www2.hm.com'
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_men_categories(self):
        try:
            url = 'https://www2.hm.com/en_us/men/products/view-all.html'
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for category links in the navigation
            categories = [
                "https://www2.hm.com/en_us/men/products/hoodies-sweatshirts.html",
                "https://www2.hm.com/en_us/men/products/hoodies-sweatshirts/hoodies.html",
                "https://www2.hm.com/en_us/men/products/hoodies-sweatshirts/printed.html",
                "https://www2.hm.com/en_us/men/products/hoodies-sweatshirts/sweatshirts.html",
                "https://www2.hm.com/en_us/men/products/pants.html",
                "https://www2.hm.com/en_us/men/products/pants/dress-pants.html",
                "https://www2.hm.com/en_us/men/products/pants/cargo-pants.html",
                "https://www2.hm.com/en_us/men/products/pants/joggers.html",
                "https://www2.hm.com/en_us/men/products/pants/linen.html",
                "https://www2.hm.com/en_us/men/products/pants/chinos.html",
                "https://www2.hm.com/en_us/men/products/shirts.html",
                "https://www2.hm.com/en_us/men/products/shirts/short-sleeve-shirts.html",
                "https://www2.hm.com/en_us/men/products/shirts/casual.html",
                "https://www2.hm.com/en_us/men/products/shirts/dressed.html",
                "https://www2.hm.com/en_us/men/products/shirts/dressed/slim-fit.html"
                "https://www2.hm.com/en_us/men/products/shirts/dressed/regular-fit.html"
                "https://www2.hm.com/en_us/men/products/shirts/linen.html",
                "https://www2.hm.com/en_us/men/products/shirts/long-sleeve.html",
                "https://www2.hm.com/en_us/men/products/shirts/flannel.html",
                "https://www2.hm.com/en_us/men/products/shirts/shirt-jackets.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/pea-coats-trenches.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/jackets.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/shirt-jackets.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/bomber-jackets.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/denim-jackets.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/puffer.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/windbreaker-jackets.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/puffer-vests.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/parkas.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/fleece-and-teddy.html",
                "https://www2.hm.com/en_us/men/products/jackets-coats/waterproof.html",
                "https://www2.hm.com/en_us/men/products/cardigans-sweaters.html",
                "https://www2.hm.com/en_us/men/products/cardigans-sweaters/cardigans.html",
                "https://www2.hm.com/en_us/men/products/cardigans-sweaters/sweaters.html",
                "https://www2.hm.com/en_us/men/products/cardigans-sweaters/turtleneck-sweaters.html",
                "https://www2.hm.com/en_us/men/products/jeans.html",
                "https://www2.hm.com/en_us/men/products/jeans/loose.html",
                "https://www2.hm.com/en_us/men/products/jeans/slim.html",
                "https://www2.hm.com/en_us/men/products/jeans/regular-fit.html",
                "https://www2.hm.com/en_us/men/products/jeans/relaxed.html",
                "https://www2.hm.com/en_us/men/products/jeans/joggers.html",
                "https://www2.hm.com/en_us/men/products/t-shirts-tank-tops.html",
                "https://www2.hm.com/en_us/men/products/t-shirts-tank-tops/graphic-printed-t-shirts.html",
                "https://www2.hm.com/en_us/men/products/t-shirts-tank-tops/short-sleeves.html",
                "https://www2.hm.com/en_us/men/products/t-shirts-tank-tops/long-sleeves.html",
                "https://www2.hm.com/en_us/men/products/t-shirts-tank-tops/tanks.html",
                "https://www2.hm.com/en_us/men/products/t-shirts-tank-tops/basics.html",
                "https://www2.hm.com/en_us/men/products/t-shirts-tank-tops/multipacks.html",
                "https://www2.hm.com/en_us/men/products/suits-blazers.html",
                "https://www2.hm.com/en_us/men/products/suits-blazers/suits.html",
                "https://www2.hm.com/en_us/men/products/suits-blazers/blazers.html",
                "https://www2.hm.com/en_us/men/products/suits-blazers/dress-pants.html",
                "https://www2.hm.com/en_us/men/products/suits-blazers/waistcoats.html",
                "https://www2.hm.com/en_us/men/products/polos.html",
                "https://www2.hm.com/en_us/men/products/premium-selection.html",
                "https://www2.hm.com/en_us/men/products/premium-selection/t-shirts.html",
                "https://www2.hm.com/en_us/men/products/premium-selection/trousers.html",
                "https://www2.hm.com/en_us/men/products/premium-selection/shoes.html",
                "https://www2.hm.com/en_us/men/products/premium-selection/shirts.html",
                "https://www2.hm.com/en_us/men/products/premium-selection/jackets-coats.html",
                "https://www2.hm.com/en_us/men/products/premium-selection/cardigans-sweaters.html",
                "https://www2.hm.com/en_us/men/products/premium-selection/accessories.html",
                "https://www2.hm.com/en_us/men/products/socks.html",
                "https://www2.hm.com/en_us/men/products/nightwear-loungewear.html",
                "https://www2.hm.com/en_us/men/products/swim-wear-trunks.html",
                "https://www2.hm.com/en_us/men/products/care-products.html",
                "https://www2.hm.com/en_us/men/shoes/sneakers.html",
                "https://www2.hm.com/en_us/men/shoes/loafers.html",
                "https://www2.hm.com/en_us/men/shoes/dress-shoes.html",
                "https://www2.hm.com/en_us/men/shoes/boots.html",
                "https://www2.hm.com/en_us/men/shoes/sandals.html",
                "https://www2.hm.com/en_us/men/shoes/slippers.html",
                "https://www2.hm.com/en_us/men/accessories/view-all.html",
                "https://www2.hm.com/en_us/men/accessories/jewelry.html",
                "https://www2.hm.com/en_us/men/accessories/hats-caps.html",
                "https://www2.hm.com/en_us/men/accessories/bags.html",
                "https://www2.hm.com/en_us/men/accessories/sunglasses.html",
                "https://www2.hm.com/en_us/men/accessories/belts-and-suspenders.html",
                "https://www2.hm.com/en_us/men/accessories/ties-bow-ties-handkerchiefs.html",
                "https://www2.hm.com/en_us/men/accessories/scarves.html",
                "https://www2.hm.com/en_us/men/accessories/gloves.html",
            ]
            
            logger.info(f"Found {len(categories)} categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []

    def get_product_urls(self, category_url):
        try:
            logger.info(f"Fetching category page: {category_url}")
            response = self.session.get(category_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all product links
            product_urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'productpage' in href:
                    full_url = self.base_url + href if href.startswith('/') else href
                    if full_url not in product_urls:
                        product_urls.append(full_url)
                        
            logger.info(f"Found {len(product_urls)} products in category")
            return product_urls

        except Exception as e:
            logger.error(f"Error fetching category page: {e}")
            return []

    def scrape_product(self, url):
        try:
            time.sleep(1)
            logger.info(f"Scraping product: {url}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get JSON-LD data
            script = soup.find('script', {'type': 'application/ld+json'})
            if not script or not script.string:
                logger.warning(f"No JSON-LD data found for {url}")
                return None
                
            product_data = json.loads(script.string)
            
            # Get images (limit to 2)
            images = []
            if isinstance(product_data.get('image'), list):
                images = product_data['image']
            elif product_data.get('image'):
                images = [product_data['image']]

            # Get colors
            colors = []
            color = product_data.get('color')
            if color:
                colors = [c.strip() for c in color.split('/')]

            # Get price
            price = None
            currency = None
            if product_data.get('offers'):
                if isinstance(product_data['offers'], list):
                    offer = product_data['offers'][0]
                else:
                    offer = product_data['offers']
                price = offer.get('price')
                currency = offer.get('priceCurrency')

            product_info = {
                'url': url,
                'name': product_data.get('name'),
                'brand': product_data.get('brand', {}).get('name'),
                'description': product_data.get('description'),
                'colors': colors,
                'sku': product_data.get('sku'),
                'price': price,
                'currency': currency,
                'images': images,
                'category': product_data.get('category', {}).get('name')
            }
            
            logger.info(f"Successfully scraped: {product_info['name']}")
            return product_info

        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None

def main():
    scraper = HMScraper()
    
    # Get men's categories
    categories = scraper.get_men_categories()
    logger.info(f"Starting to scrape {len(categories)} categories")
    
    all_products = []
    for category_url in categories:
        logger.info(f"\nProcessing category: {category_url}")
        
        # Get all product URLs for this category
        product_urls = scraper.get_product_urls(category_url)
        
        # Scrape each product
        for url in product_urls:
            product_data = scraper.scrape_product(url)
            if product_data:
                all_products.append(product_data)
                
                # Save progress every 10 products
                if len(all_products) % 10 == 0:
                    with open('hm_men_products_progress.json', 'w', encoding='utf-8') as f:
                        json.dump(all_products, f, indent=2, ensure_ascii=False)
                    logger.info(f"Progress saved: {len(all_products)} products")
    
    # Save final results
    if all_products:
        with open('hm_men_products_final.json', 'w', encoding='utf-8') as f:
            json.dump(all_products, f, indent=2, ensure_ascii=False)
        logger.info(f"\nFinished! Saved {len(all_products)} products to hm_men_products_final.json")
    else:
        logger.error("No products were successfully scraped")

if __name__ == "__main__":
    main()