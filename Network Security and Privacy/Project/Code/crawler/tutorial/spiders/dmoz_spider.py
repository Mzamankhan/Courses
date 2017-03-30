import scrapy

class DmozSpider(scrapy.Spider):
    name = "dmoz"
    #allowed_domains = ["dmoz.org"]
    start_urls = [
        "https://whisper.sh/"
    ]

    def parse(self, response):
        filename = response.url.split("/")[-2] + '.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]/div[@class="grid-item"]').extract())) #extract and write div
            divs=response.xpath('//body//div[@id="popular-whispers-container"]/div[@class="grid-item"]').extract() #extract div in variable
            imgsrc=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()
            content=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//meta/@content').extract()
            likes=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/div[@class="whisper-meta"]//ul//li/text()').extract()
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()))
            #f.write('\n'.join(map(str,divs))) #write extracted div in file divided by newline
            for i in range(len(divs)):
            	f.write(imgsrc[i]+'###')
            # ///// strip new lines from content and write to file
                contenttemp=str(content[i])
            	contenttemp=contenttemp.replace('\n', ' ').replace('\r', '')

            	f.write(contenttemp+'###')
            	f.write(likes[i]+'\t')
            	f.write(likes[i+1]+'\n\n')#+' '+content[i]+' '+likes[i]+' '+like[i+1]+'\n')
            
          


            #for d in divs:
            #	f.write(d+'\n')
           	#f.write('\n')
			
         