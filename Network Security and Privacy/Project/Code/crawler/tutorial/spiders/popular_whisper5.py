import scrapy

class DmozSpider(scrapy.Spider):
    name = "popular5"
    #allowed_domains = ["dmoz.org"]
    start_urls = [
      "https://whisper.sh/whispers/popular?wid=052cebbcc37f8ac8202c047fef744b3e643834"
    ]

    def parse(self, response):
        filename = response.url.split("?")[-1] + '.html'
        #filename='pop6'
        with open(filename, 'wb') as f:
            #f.write(response.body)
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]/div[@class="grid-item"]').extract())) #extract and write div
            divs=response.xpath('//div[@class="grid-item"]').extract()#extract div in variable
            #imgsrc=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()
            pop_content=response.xpath('//div[@class="grid-item"]//a[@class="whisper-img ga"]//meta/@content').extract()
            pop_likes=response.xpath('//div[@class="grid-item"]//div[@class="whisper-meta"]//ul/li/text()').extract()
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()))
            #f.write('\n'.join(map(str,divs))) #write extracted div in file divided by newline
            for i in range(len(divs)):
            	#f.write(imgsrc[i]+'###')
            # ///// strip new lines from content and write to file
                contenttemp=str(pop_content[i])
            	contenttemp=contenttemp.replace('\n', ' ').replace('\r', '')

            	f.write(contenttemp+'###')
            	f.write(pop_likes[i]+'###')
            	f.write(pop_likes[i+1]+'\n')#+' '+content[i]+' '+likes[i]+' '+like[i+1]+'\n')
            
          


            #for d in divs:
            #	f.write(d+'\n')
           	#f.write('\n')
			
         