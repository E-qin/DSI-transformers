# encoding=utf-8
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, BertTokenizer
import torch
import os
import json
import jieba
from tqdm import tqdm
from gensim.summarization.summarizer import summarize
import numpy as np
import faiss  
# text = "Python（英國發音：/ˈpaɪθən/ 美國發音：/ˈpaɪθɑːn/），是一种广泛使用的解释型、高级和通用的编程语言。Python支持多种编程范型，包括函数式、指令式、反射式、结构化和面向对象编程。它拥有动态类型系统和垃圾回收功能，能够自动管理内存使用，并且其本身拥有一个巨大而广泛的标准库。它的语言结构以及面向对象的方法旨在帮助程序员为小型的和大型的项目编写清晰的、合乎逻辑的代码。"

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def create_queries(para, mode):
    # QWC: hugging face model name: "doc2query/msmarco-chinese-mt5-base-v1"
    if mode == 'doc2query':
        model_name = '/home/weijie_yu/pretrain_models/doc2query-chinese'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # QWC: https://github.com/ZhuiyiTechnology/t5-pegasus
    if mode == 't5pegasus':
        model_name = '/home/weijie_yu/pretrain_models/T5_PEGASUS'
        tokenizer = T5PegasusTokenizer.from_pretrained(model_name)
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer)) 
    input_ids = tokenizer.encode(para, return_tensors='pt')

    with torch.no_grad():
        # Here we use top_k / top_k random sampling. It generates more diverse queries, but of lower quality
        sampling_outputs = model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            top_k=10, 
            num_return_sequences=5
            )
        
        # Here we use Beam-search. It generates better quality queries, but with less diversity
        beam_outputs = model.generate(
            input_ids=input_ids, 
            max_length=64, 
            num_beams=5, 
            no_repeat_ngram_size=2, 
            num_return_sequences=5, 
            early_stopping=True
        )


    # print("Paragraph:")
    # print(para)
    
    print("\nBeam Outputs:")
    beam_out = []
    for i in range(len(beam_outputs)):
        query = tokenizer.decode(beam_outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')
        # beam_out.append(query)

    print("\nSampling Outputs:")
    sample_out = [] 
    for i in range(len(sampling_outputs)):
        query = tokenizer.decode(sampling_outputs[i], skip_special_tokens=True)
        print(f'{i + 1}: {query}')
        # sample_out.append(query)

    # return beam_out, sample_out


def lecard_inverted_dict(path_in_1, path_in_2, path_out):
    # 将lecard原始数据（输入）构建成{文件名:[适用罪名]}的形式（输出）
    path_list = [path_in_1, path_in_2]
    filename_law_dict = {}
    for path in path_list:
        with open(path, 'r') as r:
            law_filename_dict = json.load(r)            
            for law, filename_list in law_filename_dict.items():
                for filename in filename_list:
                    if filename not in filename_law_dict:
                        filename_law_dict[filename] = []
                    filename_law_dict[filename].append(law)
    with open(path_out, 'w') as w:
        dict_to_write = json.dumps(filename_law_dict, ensure_ascii=False)
        w.write(dict_to_write) 


def generate_query_lecard(path_prefix, path_in, path_out):
    # 输入是lecard_inverted_dict的输出
    # 输出是{文件名:{text:'', query:'', crime:[]}}
    output_dict = {}
    with open(path_in, 'r') as r:
        with open(path_out, 'w') as w:
            finename_crime_dict = json.load(r)
            for filename, crime in tqdm(finename_crime_dict.items()):
                abs_path = path_prefix + filename
                # abs_name是文件夹名称，包含一个或多个文件
                # for case in os.listdir(abs_name):
                with open(abs_path, 'r') as r:
                    case_dict = json.load(r)
                    # "ajId" "ajName" "ajjbqk" "cpfxgc" "pjjg" "qw" "writId" "writName"
                    writId = case_dict["writId"]
                    output_dict[writId] = {}
                    for key,val in case_dict.items():
                        if key == 'writId':
                            continue
                        elif key == 'ajjbqk':
                            beam_out, sample_out = create_queries(val)
                            output_dict[writId]['query_beam'] = beam_out
                            output_dict[writId]['query_topp'] = sample_out
                        else:
                            output_dict[writId][key] = val
                    output_dict[writId]['crime'] = crime
                    print(output_dict)
    json_to_write = json.dumps(output_dict, ensure_ascii=False)
    w.write(json_to_write)





if __name__ == '__main__':
    # fact = "广东省湛江市人民检察院指控：2014年5月22日9时许，被告人庄彪叫经营精品的朋友万某开小车搭载其从湛江到广州，承诺支付1000元作为路费、加油费，万某表示同意并邀请吴某与其一起去广州采购精品，吴某表示同意。 2014年5月22日14时许，万某驾车牌号为粤G×××××的黑色本田雅阁小轿车搭载吴某从湛江市霞山区城市广场出发，去到湛江市××××道城市美林住宅小区附近等庄彪。15时许，庄彪从住处提着一个黑色长方形皮革行李袋出来，坐上万某驾驶的小轿车后座，途中庄彪告诉万某要去东莞。公安机关根据线索掌握了驾乘粤G×××××的黑色本田雅阁小轿车的人员涉嫌毒品犯罪后，公安人员跟踪尾随该小轿车，并事先在湛江市××××区茂湛高速公路官渡收费站设卡守候，当庄彪等人的小车行驶至官渡收费站时，被公安人员拦停并进行搜查，从庄彪的黑色长方形皮革行李袋里查获疑似毒品冰毒2包、枪形物体1支、子弹7发。经检验鉴定：疑似毒品冰毒2包共净重1839克，检见甲基苯丙胺成份；枪形物体是能以火药为动力有效发射金属弹丸的12号猎枪，7发子弹中5发是自制猎枪弹、2发是12号猎枪弹。当晚22时许，公安机关对庄彪的住处进行搜查，从庄彪住处查获疑似毒品红色颗粒7小包、疑似毒品冰毒1小包、枪形物体1支、子弹18发。经检验鉴定：疑似毒品红色颗粒7小包共净重121.64克，检见甲基苯丙胺成份；疑似毒品冰毒1小包共净重4.16克，检见甲基苯丙胺成份；枪形物体是能以火药为动力有效发射金属弹丸的12号猎枪，18发子弹中3发是自制猎枪弹、15发是12号猎枪弹。 针对上述指控，公诉机关当庭提供了相应的证据。据以上事实和证据，公诉机关指控被告人庄彪无视国家法律，违反国家对毒品和枪支弹药的管理规定，非法运输甲基苯丙胺1839克，非法持有甲基苯丙胺125.8克，非法持有12号猎枪2支、自制猎枪弹8发、12号猎枪弹17发，其行为已触犯了《中华人民共和国刑法》第三百四十七条、第三百四十八条、第一百二十八条之规定，犯罪事实清楚，证据确实、充分，应当以运输毒品罪、非法持有毒品罪、非法持有枪支罪追究其刑事责任。且庄彪属于累犯、毒品犯罪再犯，依法应从重处罚。根据《中华人民共和国刑事诉讼法》第一百七十二条之规定，提请本院依法判处。 被告人庄彪在庭审时辩解称：其自己随身携带的毒品是方便自己吸食，其行为不构成运输毒品罪；侦查机关没有将查获的1839克甲基苯丙胺的含量鉴定意见告知其，要求重新作鉴定。 被告人庄彪的辩护人提出如下辩护意见：1、被告人庄彪为满足自身吸毒的需要而携带、窝藏毒品的行为，不应认定为运输毒品。（1）、本案无证据显示庄彪携带1839克毒品甲基苯丙胺的行为系用于自身吸食外的任何非法用途。（2）、从本案的证据可以推断：庄彪吸毒时间较长、且吸食量很大，其携带大量毒品准备长时间出差自用、窝藏的行为是符合常理的。2、公诉机关对庄彪属于累犯、毒品再犯的指控属于法条竞合，本案应仅适用毒品再犯的规定。3、庄彪能坦白、认罪态度较好，对于司法机关的侦查、审查起诉等行为始终予以配合，依法可从轻处罚。4、鉴定机关对查获的1839克甲基苯丙胺含量作出鉴定意见，侦查机关没有将用作证据的鉴定意见告知庄彪，程序违法，要求对此进行重新鉴定。 经审理查明：2014年5月22日9时许，被告人庄彪叫经营精品的朋友万某开小车搭载其从湛江到广州，承诺支付1000元作为路费、加油费，万某表示同意并邀请吴某与其一起去广州采购精品，吴某表示同意。 当天14时许，万某驾驶车牌号为粤G×××××的黑色本田雅阁小轿车搭载吴某从湛江市霞山区城市广场出发，去到湛江市××××道城市美林住宅小区附近等庄彪。15时许，庄彪从住处提着一个黑色长方形皮革行李袋出来，坐上万某驾驶的小轿车后座，途中庄彪告诉万某要去东莞。公安机关根据线索掌握了驾乘粤G×××××的黑色本田雅阁小轿车的人员涉嫌毒品犯罪后，派员跟踪尾随该小轿车，并事先在湛江市××××区茂湛高速公路官渡收费站设卡守候，当庄彪等人的小车行驶至官渡收费站时，被公安人员拦停并进行搜查，公安人员从庄彪放置在粤G×××××小车后座地板上的黑色长方形皮革行李袋里查获疑似毒品冰毒2包、枪形物体1支、子弹7发。经检验鉴定：疑似毒品冰毒2包共净重1839克，检见甲基苯丙胺成份，含量为41.4％；枪形物体是能以火药为动力有效发射金属弹丸的12号猎枪，7发子弹中5发是自制猎枪弹、2发是12号猎枪弹。当天22时许，公安机关对位于湛江市人民大道城市美林10栋1座1805房庄彪的租住处进行搜查，查获疑似毒品红色颗粒7小包、疑似毒品冰毒1小包、枪形物体1支、子弹18发。经检验鉴定：疑似毒品红色颗粒7小包共净重121.64克，疑似毒品冰毒1小包共净重4.16克，均检见甲基苯丙胺成份；枪形物体是能以火药为动力有效发射金属弹丸的12号猎枪，18发子弹中3发自制猎枪弹、15发是12号猎枪弹。另查明，2014年5月23日，经对庄彪的尿液现场检测做甲基苯丙胺筛选试验，结果呈阳性。 上述事实，有公诉机关提交，并经法庭质证、认证的下列证据予以证明： 一、物证、书证 1、受案登记表、立案决定书、到案经过、犯罪嫌疑人归案情况说明，主要内容：公安机关侦破本案及抓获被告人庄彪的情况。 2、常住人口信息：被告人庄彪的基本身份情况，其作案时已年满十八周岁。 3、刑事判决书、刑满释放证明书：被告人庄彪因犯非法持有毒品罪，于2012年9月28日被湛江市霞山区人民法院判处有期徒刑九个月，并处罚金人民币五千元，于2013年2月17日刑满释放。 4、搜查证、搜查笔录、扣押决定书：公安机关在粤G×××××小轿车发现疑似毒品冰毒2袋，枪形物体1支、子弹7颗。公安机关在湛江市人民大道城市美林10栋1座1805房发现疑似毒品麻果7小包、疑似毒品冰毒1小包、枪形物体1支、子弹18颗。公安机关对上述疑似毒品、枪形物体、子弹予以扣押。同时公安机关扣押了被告人庄彪手机4台。被告人庄彪对公安机关扣押的上述疑似毒品、枪形物体、子弹进行辨认、确认。证人万某、吴某对被告人庄彪在粤G×××××小轿车被扣押的疑似毒品、枪形物体、子弹进行辨认、确认。证人谢某甲对被告人庄彪在湛江市人民大道城市美林10栋1座1805房被扣押的疑似毒品、枪形物体、子弹进行辨认、确认。 5、行政处罚决定书：被告人庄彪因吸毒于2014年5月23日被湛江市公安局霞山分局处以拘留十五日。被告人庄彪因赌博及吸毒于2014年3月19日被湛江市公安局霞山分局合并处以行政拘留二十日并处罚款五百元。 6、现场检测报告书：2014年5月23日，经对被告人庄彪的尿液现场检测做甲基苯丙胺筛选试验，结果呈阳性。 7、湛江市港区人民医院体格检查表、入所体检表：被告人庄彪入看守所前进行体检，身体未发现异常。 8、中国移动通信客户详单：被告人庄彪于2014年5月22日与万某有7次通话记录。 二、证人证言及辨认笔录 1、证人万某的证言，主要内容：2014年5月22日9时许，我发现我手机凌晨有个未接电话，是庄彪打来的，于是我拨打回去问何事，庄彪问我车是否有空，要我当天14、15时开车到城市美林接他后送他到广州，付我1000元作为加油费及路费。我说可以。当天13时许，吴某来到找我，我向吴某提出由其在广州带我进点精品货，吴某同意。后来庄彪打手机问我是否可以出车，我说准备好了。14时许，我开粤G×××××小汽车搭吴某到城市美林门口路边等庄彪，一会庄彪拿着一个黑色长方形皮革提袋过来坐上车后排，庄彪上车后对我说到虎门大桥，我开车沿人民大道开往坡头区官渡高速入口。16时许，我们到达坡头区茂湛高速公路官渡收费站入口处，我停车取卡时被便衣警察拦截检查，我们三人被抓获，警察当场在我小车后排庄彪的黑色提袋中检查搜缴到2包白色晶体和1支猎枪、7颗子弹。经警察当面称重，2包白色晶体中1包毛重1049.53克，另1包毛重803.54克。 经对混杂照片进行辨认，万某辨认出被告人庄彪。 2、证人吴某的证言，与万某的证言内容一致。 经对混杂照片进行辨认，吴某辨认出被告人庄彪。 3、证人谢某甲的证言，主要内容：我与我丈夫庄彪居住在湛江市人民大道城市美林10栋1座1805房，2014年5月22日22时许，民警在该房靠近大厅的房间衣柜中间发现一可疑羽毛球拍袋，打开后发现内有一支褐色的猎枪，在球拍袋旁边发现一个黄色布袋，打开后发现内藏猎枪子弹18颗。衣柜中间的位置发现一小包可疑毒品的白色晶体。警察在大厅后面的房间的衣柜的顶层位置发现一个外写有“香港飞大珠宝”的红色纸袋，纸袋内装有7小包蓝色塑料袋，打开后都是红色及绿色的可疑颗粒，警察依法将此缴获。被缴获的枪支弹药和可疑毒品是庄彪的。5月22日14时许，庄彪离开家时拿着一个袋子。 4、证人谢某乙的证言，主要内容：2013年10月，其将湛江市人民大道城市美林10栋1座1805房租给谢某甲和庄彪一家住。 三、被告人的供述和辩解、辨认笔录 被告人庄彪的供述和辩解，主要内容：2014年5月22日上午，我用手机联系“济公”（万某）叫其开车搭我到广州出差，我提出给其1000元作路费及加油费，万某同意。当天14时许，万某开粤G×××××黑色本田雅阁小车到霞山区××××道城市美林接我，我拿着黑色提袋上车就出发了。我们的车沿人民大道开往坡头区官渡高速入口，当时万某开车，他的朋友已经坐在车前座。16时许，我们到达坡头区茂湛高速公路官渡入口处，停车取卡时被警察拦截检查，我们三人被抓获，警察当场在小车后排我拿的黑色提袋中查缴到2袋透明晶体（冰毒）、1支雷明登枪及7颗子弹，还有1台粉红色的诺基亚手机。我本来想到深圳、广州、虎门等地看看有什么生意做，后来决定到广州。我带枪是为了出差防身。公安机关在我位于霞山区人民大道城市美林10栋1座1805房搜查并扣押到7小包毒品麻果、1小包冰毒、1支雷明登枪及18发子弹、3台手机。 我被扣押到的毒品是前段时间在霞山区火车南站附近向“老鬼”购买的，2支枪及子弹是“广西仔”于2年前在机场路卖给我的，共2万元。我购买毒品是方便自己吸食，买枪支弹药是为了搞养殖时好防身。 我于2010年开始吸食毒品冰毒、麻果，现已吸毒成瘾。我每天都要吸食5－6克的冰毒和20－30粒的麻果。 经对混杂照片进行辨认，庄彪辨认出“济公”系万某。 四、鉴定意见 1、湛江市公安司法鉴定中心粤湛公（司）鉴（化）字（2014）641、642、1573号毒品检验鉴定报告：公安机关在粤G×××××小轿车上搜出可疑毒品2包共净重1839克，均检见甲基苯丙胺成份，含量为41.4％。公安机关在湛江市人民大道城市美林10栋1座1805房庄彪的住宅搜出的丸状固体的可疑毒品7小包净重121.64克，白色晶体的可疑毒品1小包净重4.16克，均检见甲基苯丙胺成份。 2、湛江市公安司法鉴定中心粤湛公（司）鉴（痕）字（2014）84、85号枪支弹药检验报告：公安机关在粤G×××××小轿车上及湛江市人民大道城市美林10栋1座1805房庄彪的住宅搜出的枪形物体2件均为能以火药为动力有效发射金属弹丸的12号猎枪；在粤G×××××小轿车上搜出的子弹7发中有5发是自制猎枪弹，有2发是12号猎枪弹；在湛江市人民大道城市美林10栋1座1805房庄彪的住宅搜出的18发子弹中有3发是自制猎枪弹，有15发是12号猎枪弹。 五、勘验、检查笔录 现场勘验、检查笔录、示意图、照片、提取痕迹、物证登记表：第一现场位于湛江市××××区广湛高速官渡入口收费站，该高速入口分为6条车道，其中由东向西数第4车道收费亭对应位置停有一辆悬挂车牌号为粤G×××××的黑色本田雅阁牌小轿车，公安人员在该车后座地板上的黑色袋里提取枪形物体1支、子弹7颗、透明塑料袋包装的白色晶体2袋。被告人庄彪、证人万某、吴某对现场及粤G×××××的黑色本田雅阁牌小轿车进行辨认、确认。第二现场位于湛江市人民大道城市美林10栋1座1805房，公安人员在该房西南角卧室衣柜北格提取枪形物体1支、子弹18颗，在该衣柜中格提取透明塑料袋包装的白色晶体1袋，在该房西北角卧室衣柜西格上层提取蓝色塑料袋包装的红色颗粒7袋。 六、辨认材料，被告人庄彪、证人万某、吴某对庄彪拿黑色手提袋在城市美林侧门上万某驾驶的黑色小车的截图及进行指认。 七、现场监控视频，主要内容：被告人庄彪从城市美林小区出来，坐上万某驾驶的粤G×××××小车离开及在官渡收费站被拦截抓获的行为情况。 针对被告人庄彪及其辩护人提出的辩解、辩护意见，根据本案的事实、证据和相关法律规定，本院评判如下： 1、对于被告人庄彪及其辩护人所提被告人庄彪随身携带的毒品是为自己吸食，其行为不构成运输毒品罪的辩解、辩护意见。 经查，本案的证据证明，被告人庄彪虽然系吸毒者，但其系在运输毒品过程中被当场查获数量大、明显超出其个人正常吸食量的毒品，故其行为应依法认定为运输毒品罪。庄彪及其辩护人的前述辩护、辩解意见不能成立，本院不予采纳。 2、对于被告人庄彪及其辩护人所提要求对本案涉案的1839克毒品甲基苯丙胺的含量作重新鉴定的辩解、辩护意见。 经查，湛江市公安司法鉴定中心对于从被告人庄彪处缴获1839克疑似毒品进行了检验，证实检出甲基苯丙胺成分，含量为41.4％。该检验报告书客观、真实地反映了检验对象所含有的毒品种类和含量。虽然公安机关没有将毒品含量的检验报告书告知被告人庄彪，但公诉机关在庭审举证时已将该检验报告书给被告人庄彪及其辩护人出示和宣读，庄彪及其辩护人已了解该检验报告的内容，并对此发表意见。上述鉴定意见已经当庭出示、质证等法庭调查程序查证属实，可以作为认定本案事实的证据。故庄彪及其辩护人所提前述的辩解、辩护意见与查明的事实和证据不符，且没有证据支持，本院不予支持。 3、对于辩护人所提公诉机关对被告人庄彪属于累犯、毒品再犯的指控属于法条竞合，本案应仅适用毒品再犯的规定的辩护意见。 经查，最高人民法院要求各地法院参照执行的《全国部分法院审理毒品犯罪案件工作座谈会纪要》规定，“对同时构成累犯和毒品再犯的被告人，应当同时引用刑法关于累犯和毒品再犯的条款从重处罚。”故辩护人所提的前述辩护意见不能成立，本院不予采纳。 4、对于辩护人所提被告人庄彪认罪态度好的辩护意见与查明的事实相符，本院予以采纳。 "
    # # create_queries(fact, 't5pegasus')
    # print(summarize(fact))
    # PATH_1 = '/home/weijie_yu/dataset/legal/lecard/LeCaRD-main/data/corpus/common_charge.json'
    # PATH_2 = '/home/weijie_yu/dataset/legal/lecard/LeCaRD-main/data/corpus/controversial_charge.json'
    # PATH_OUT = '/home/weijie_yu/dataset/legal/lecard/LeCaRD-main/data4lm/filename_law_dict.json'
    # lecard_inverted_dict(PATH_1, PATH_2, PATH_OUT)

    # path_prefix = '/home/weijie_yu/dataset/legal/lecard/documents/'
    # path_in = '/home/weijie_yu/dataset/legal/lecard/LeCaRD-main/data4lm/filename_law_dict.json'
    # path_out = '/home/weijie_yu/dataset/legal/lecard/LeCaRD-main/data4lm/case_query.json'
    # generate_query_lecard(path_prefix, path_in, path_out)
    pass
