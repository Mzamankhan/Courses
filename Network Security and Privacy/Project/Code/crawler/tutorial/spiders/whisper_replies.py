import scrapy

class DmozSpider(scrapy.Spider):
    name = "replies"
    #allowed_domains = ["dmoz.org"]
    start_urls = [
        "https://whisper.sh/"
        #"https://whisper.sh/whispers/popular?wid=052caf88c3a99eb4788dcdc1f77015d8d869be"
        #"https://whisper.sh/whispers/popular?wid=052caeb80237af1d76bb956d76e0ed9e60474b"
        #"https://whisper.sh/whispers/popular?wid=052caea0e75a6b5dafb46bfe53159035332077"
        #"https://whisper.sh/whispers/popular?wid=052c7cd596971cd93b3275c2ea36c8d6476803"
        #"https://whisper.sh/whispers/popular?wid=052caf79cddf32b3cddb4ecdb4c30f2f41d237"
        #"https://whisper.sh/whispers/popular?wid=052c9f38debb39df02f3c2e4eb12509d9b5e2f"
        #"https://whisper.sh/whispers/popular?wid=052cb1c55ddbc5c951fe03ccd1cc08058fabd6"
        #"https://whisper.sh/whispers/popular?wid=052c8c336a3dbc7362cb668094b55824498c95"
    ]
    

    def parse(self, response):
        filename = response.url.split("/")[-2] + '.html'
        #filename='testfile.html'
        with open(filename, 'wb') as f:
            #f.write(response.body)
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]/div[@class="grid-item"]').extract())) #extract and write div
            #divs=response.xpath('//body//div[@id="popular-whispers-container"]/div[@class="grid-item"]').extract() #extract div in variable
            imgsrc=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()
            content=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//meta/@content').extract()
            likes=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/div[@class="whisper-meta"]//ul//li/text()').extract()
            #f.write(str(response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]//img/@src').extract()))
            urls=response.xpath('//body//div[@id="popular-whispers-container"]//div[@class="grid-item"]/a[@class="whisper-img ga"]/@href').extract()
            #f.write('\n'.join(map(str,divs))) #write extracted div in file divided by newline
            for i in range(len(imgsrc)):
                f.write(imgsrc[i]+'###')
            # ///// strip new lines from content and write to file
                contenttemp=str(content[i])
                contenttemp=contenttemp.replace('\n', ' ').replace('\r', '')
 
                f.write(contenttemp+'###')
                f.write(likes[i]+'###')
                f.write(likes[i+1]+'\n\n')#+' '+content[i]+' '+likes[i]+' '+like[i+1]+'\n')
                built_url='https://whisper.sh'+str(urls[i])
                f.write(built_url+'\n\n\n')
                yield scrapy.Request(built_url, callback=self.parse_replies)


    def parse_replies(self,response):
        # /// get main post, replace all new lines with space
        mainpost=response.xpath('//head//title/text()').extract()
        main_post=str(mainpost)
        main_post=main_post.replace('\n', ' ').replace('\r', '')

        # /// get reply content, replace new lines with space
        reply_content=response.xpath('//body//div[@class="whispers-more-row-inner"]//div[@class="col-xs-6"]//div[@class="whisper-cont ga"]//a[@class="whisper-img"]//meta/@content').extract()
       

        # /// get image links
        reply_image=response.xpath('//body//div[@class="whispers-more-row-inner"]//div[@class="col-xs-6"]//a[@class="whisper-img"]/@src').extract()

        # /// get likes and replies

        reply_likes=response.xpath('//body//div[@class="whispers-more-row-inner"]//div[@class="col-xs-6"]//div[@class="whisper-cont ga"]//div[@class="whisper-meta"]//ul//li/text()').extract()

        filename1=main_post+'.html'
        with open(filename1, 'wb') as f1:
            f.write('MainPost###'+main_post+'\n')
            for rc in range(len(reply_content)):
                 reply_content1=str(reply_content[rc])
                 reply_content1=reply_content1.replace('\n', ' ').replace('\r', '')
                 f.write(reply_content1+'###')
                 f.write(reply_likes[rc]+'###')
                 f.write(reply_likes[rc+1]+'\n\n')







            
            