from langchain_text_splitters import RecursiveCharacterTextSplitter

# 模拟一份"员工手册"，里面有三个不同主题
text = """一、请假制度
员工请假需提前在OA系统提交申请，经直属主管审批后方可休假。事假为无薪假期，每次事假原则上不超过三天。病假需提供二级以上医院开具的诊断证明，病假期间按当地最低工资标准的80%发放。员工连续工作满一年享有5天带薪年假，满十年享有10天年假，年假需提前一周申请。
二、费用报销
差旅费需在出差结束后7个工作日内提交报销单，逾期不予受理。市内出租车票需在发票背面注明起止地点和事由。团队聚餐人均标准不超过100元，超出部分由参与者自行承担。所有报销须附正规发票，收据和截图无效。
三、办公设备
公司为每位新员工配备笔记本电脑一台，入职当天到IT部领取。电脑保修期内的非人为损坏由公司负责维修。员工离职时须将电脑、电源和门禁卡一并归还，如有遗失或人为损坏需照价赔偿。"""
def split_and_show(size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        length_function=len,          # 中文按"字符数"算，不是按空格分词
    )
    chunks = splitter.split_text(text)
    print(f"\n===== chunk_size={size}, overlap={overlap}，共 {len(chunks)} 块 =====")
    for i, c in enumerate(chunks):
        print(f"[块{i}] ({len(c)}字)\n{c}\n")
if __name__ == "__main__":
    split_and_show(150, 30)   # 切得碎
    split_and_show(400, 60)   # 切得粗，对比看区别
    split_and_show(50, 10)
