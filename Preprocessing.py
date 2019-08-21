import pandas as pd
import numpy as np
df1=pd.read_csv("demographic.csv", sep=',' )
df2=pd.read_csv("examination.csv", sep=',' )
df3=pd.read_csv("labs.csv", sep=',' )
df4=pd.read_csv("questionnaire.csv", sep=',' )

#Merging all subsets and creating the Dataset
df = pd.merge(df1, df2, how='inner', on='SEQN')
df.shape #(9813, 270)

df = pd.merge(df, df3, how='inner', on='SEQN')
df.shape #(9813, 693)

df = pd.merge(df, df4, how='inner', on='SEQN')
df.shape #(9813, 1645)

df = df.drop(['SDDSRVYR', 'RIDSTATR', 'RIDAGEMN','RIDRETH1','RIDEXMON','RIDEXAGM','DMDYRSUS','DMDEDUC3',
              'DMDEDUC2','DMQMILIZ', 'DMQADFC','SIAPROXY','SIAINTRP', 'FIALANG','FIAPROXY', 'FIAINTRP',
              'MIALANG', 'MIAPROXY','MIAINTRP', 'AIALANGA', 'DMDHHSIZ','DMDHHSZA','DMDHHSZB','DMDHHSZE',
              'DMDHRGND', 'DMDHRAGE', 'DMDHRBR4', 'DMDHREDU','DMDHRMAR','DMDHSEDU','WTINT2YR','WTMEC2YR',
              'SDMVPSU','SDMVSTRA','INDHHIN2','INDFMPIR','PEASCST1', 'PEASCTM1', 'PEASCCT1','BPXCHR', 'BPAARM',
              'BPACSZ','BPXPLS','BPXPULS', 'BPXPTY','BPXML1','BPAEN1', 'BPAEN2', 'BPXSY3', 'BPXDI3', 'BPAEN3', 'BPXSY4',
              'BMIWT', 'BMXRECUM','BMIRECUM', 'BMXHEAD','BMIHEAD', 'BMIHT', 'BMDBMIC', 'BMXLEG','BMILEG','BMXARML',
              'BMIARML', 'BMXARMC', 'BMIARMC', 'BMXWAIST','BMIWAIST', 'BMXSAD1', 'BMXSAD2', 'BMXSAD3', 'BMXSAD4',
              'BMDAVSAD', 'BMDSADCM','MGDEXSTS','MGD050','MGD060','MGQ070','MGQ080','MGQ090','MGQ100','MGQ110',
              'MGQ120','MGD130','MGQ90DG','MGDSEAT', 'MGAPHAND','MGATHAND','MGXH1T1E','MGXH2T1E','MGXH1T2','MGXH1T2E',
              'MGXH2T2','MGXH2T2E','OHX04TC','OHX05TC','OHX06TC','BPXDI4', 'BPAEN4','BMDSTATS',
              'MGXH1T3','MGXH1T3E','MGXH2T3','MGXH2T3E','OHDEXSTS','OHDDESTS','OHXIMP','OHX01TC','OHX02TC','OHX03TC',
              'OHX07TC','OHX08TC','OHX09TC','OHX10TC','OHX11TC','OHX12TC','OHX13TC','OHX14TC','OHX15TC',
              'OHX16TC','OHX17TC','OHX18TC','OHX19TC','OHX20TC','OHX21TC','OHX22TC','OHX23TC','OHX24TC',
              'OHX25TC','OHX26TC','OHX27TC','OHX28TC','OHX29TC','OHX30TC', 'OHX31TC','OHX32TC','OHX02CTC','OHX03CTC',
              'OHX04CTC','OHX05CTC','OHX06CTC','OHX07CTC','OHX08CTC','OHX09CTC','OHX10CTC','OHX11CTC','OHX12CTC',
              'OHX13CTC','OHX14CTC','OHX15CTC','OHX18CTC','OHX19CTC','OHX20CTC','OHX21CTC',
              'OHX22CTC','OHX23CTC','OHX24CTC','OHX25CTC','OHX26CTC','OHX27CTC','OHX28CTC','OHX29CTC','OHX30CTC',
              'OHX31CTC', 'OHX02CSC','OHX03CSC','OHX04CSC','OHX05CSC','OHX06CSC','OHX07CSC','OHX08CSC','OHX09CSC',
              'OHX10CSC','OHX11CSC','OHX12CSC','OHX13CSC','OHX14CSC','OHX15CSC','OHX18CSC',
              'OHX19CSC','OHX20CSC','OHX21CSC','OHX22CSC','OHX23CSC','OHX24CSC','OHX25CSC','OHX26CSC','OHX27CSC',
              'OHX28CSC','OHX29CSC','OHX30CSC','OHX31CSC', 'OHX02SE','OHX03SE', 'OHX04SE','OHX05SE','OHX07SE','OHX10SE',
              'OHX12SE','OHX13SE','OHX14SE','OHX15SE','OHX18SE','OHX19SE','OHX20SE','OHX21SE','OHX28SE','OHX29SE',
              'OHX30SE','OHX31SE','CSXEXSTS','CSXEXCMT','CSQ245','CSQ241','CSQ260A','CSQ260D','CSQ260G','CSQ260I',
              'CSQ260N','CSQ260M','CSQ270','CSQ450','CSQ460','CSQ470','CSQ480','CSQ490','CSXQUIPG',
              'CSXQUIPT','CSXNAPG','CSXNAPT','CSXQUISG','CSXQUIST','CSXSLTSG','CSXSLTST','CSXNASG',
              'CSXNAST','CSXTSEQ','CSXCHOOD','CSXSBOD','CSXSMKOD','CSXLEAOD','CSXSOAOD','CSXGRAOD',
              'CSXONOD','CSXNGSOD','CSXSLTRT','CSXSLTRG','CSXNART','CSXNARG','CSAEFFRT','MGDCGSZ','URXUMA',
              'URDACT','URXCRS','LBXAPB','WTSAF2YR.x','WTSA2YR','URXUAS','URXUCR',
              'URXUAS3','URDUA3LC','URXUAS5','URDUA5LC','URXUAB','URDUABLC','URXUAC','URDUACLC',
              'URXUDMA','URDUDALC','URXUMMA','WTSH2YR.x','LBDBPBSI','LBDBPBLC','LBDBCDSI','LBDBCDLC',
              'LBDTHGSI','LBDTHGLC','LBDBSESI','LBDBSELC','LBDBMNSI','LBDBMNLC','LBXIHG','LBDIHGSI','LBDIHGLC',
              'LBXBGE','LBDBGELC','LBXBGM','LBDBGMLC','URXUCL','LBDHDD','LBDHDDSI','LBXTR','LBDTRSI',
              'WTSAF2YR','LBDLDL','LBDLDLSI','LBDTCSI','LBXLYPCT','LBXMOPCT','LBXNEPCT','LBXEOPCT',
              'LBXBAPCT','LBDLYMNO','LBDMONO','LBDNENO','LBDEONO','LBDBANO','LBXHCT','LBXMCVSI','LBXMCHSI',
              'LBXMC','LBXRDW','LBXMPSI','LBXSCU','LBDSCUSI','LBXSSE','LBDSSESI','LBXSZN','LBDSZNSI',
              'LBXHCT','PHQ020','PHACOFHR','PHACOFMN','PHQ030','PHAALCHR','PHAALCMN','PHQ040','PHAGUMHR',
              'PHAGUMMN','PHQ050','PHAANTHR','PHAANTMN','PHQ060','PHASUPHR','PHASUPMN','PHAFSTHR','PHAFSTMN',
              'PHDSESN','LBDPFL','LBDWFL','LBXGH','LBXHBC','LBDHBG','LBDHD','LBXHBS','LBXHCR','LBXHCG',
              'LBDHEG','LBDHEM','LBXHE1','LBXHE2','LBDHI','ORXGH','ORXGL','ORXH06','ORXH11','ORXH16','ORXH18',
              'ORXH26','ORXH31','ORXH33','ORXH35','ORXH39','ORXH40','ORXH42','ORXH45','ORXH51','ORXH52','ORXH53',
              'ORXH54','ORXH55','ORXH56','ORXH58','ORXH59','ORXH61','ORXH62','ORXH64','ORXH66','ORXH67','ORXH68',
              'ORXH69','ORXH70','ORXH71','ORXH72','ORXH73','ORXH81','ORXH82','ORXH83','ORXH84','ORXHPC','ORXHPI',
              'ORXHPV','LBXHP2C','LBDINSI','PHAFSTHR','PHAFSTMN','URXUIO','URXUHG','URDUHGLC','URXUBA','URDUBALC',
              'URXUCD','URDUCDLC','URXUCO','URDUCOLC','URXUCS','URDUCSLC','URXUMO','URDUMOLC','URXUMN','URDUMNLC',
              'URXUPB','URDUPBLC','URXUSB','URDUSBLC','URXUSN','URDUSNLC','URXUSR','URDUSRLC','URXUTL','URDUTLLC',
              'URXUTU','URDUTULC','URXUUR','URDUURLC','LBDGLTSI','GTDSCMMN','GTDDR1MN','GTDBL2MN','GTDDR2MN','GTXDRANK',
              'PHAFSTHR','PHAFSTMN','GTDCODE','LBXGLT','URXUP8','URDUP8LC','URXNO3','URDNO3LC','URXSCN','URDSCNLC',
              'LBXPFDE','LBDPFDEL','LBXPFHS','LBDPFHSL','LBXMPAH','LBDMPAHL','LBXPFBS','LBDPFBSL','LBXPFHP',
              'LBDPFHPL','LBXPFNA','LBDPFNAL','LBXPFUA','LBDPFUAL','LBXPFDO','LBDPFDOL','URXBP3','URDBP3LC',
              'URXBPH','URDBPHLC','URXBPF','URDBPFLC','URXBPS','URDBPSLC','URXTLC','URDTLCLC','URXTRS','URDTRSLC',
              'URXBUP','URDBUPLC','URXEPB','URDEPBLC','URXMPB','URDMPBLC','URXPPB','URDPPBLC','URX14D','URD14DLC',
              'URXDCB','URDDCBLC','URXCNP','URDCNPLC','URXCOP','URDCOPLC','URXECP','URDECPLC','URXMBP','URDMBPLC',
              'URXMC1','URDMC1LC','URXMEP','URDMEPLC','URXMHH','URDMHHLC','URXMHNC','URDMCHLC','URXMHP','URDMHPLC',
              'URXMIB','URDMIBLC','URXMNP','URDMNPLC','URXMOH','URDMOHLC','URXMZP','URDMZPLC','URXP01','URDP01LC',
              'URXP02','URDP02LC','URXP03','URDP03LC','URXP04','URDP04LC','URXP06','URDP06LC','URXP10','URDP10LC',
              'URXP25','URDP25LC','LBXSAL','LBDSALSI','LBXSAPSI','LBXSASSI','LBXSATSI','LBXSBU','LBDSBUSI',
              'LBXSC3SI','LBXSCA','LBDSCASI','LBDSCHSI','LBXSCK','LBXSCLSI','LBXSCR','LBDSCRSI','LBDSGBSI',
              'LBDSGLSI','LBXSGTSI','LBXSIR','LBDSIRSI','LBXSKSI','LBXSLDSI','LBXSNASI','LBXSOSSI','LBXSPH',
              'LBDSPHSI','LBXSTB','LBDSTBSI','LBDSTPSI','LBXSTR','LBDSTRSI','LBXSUA','LBDSUASI','LBXTTG',
              'LBXEMA','URXUTRI','URXVOL1','URDFLOW1','URXVOL2','URDFLOW2','URXVOL3','URDFLOW3','LBDB12SI',
              'LBDRPCR.x','LBDRHP.x','LBDRLP.x','LBDR06.x','LBDR11.x','LBDR16.x','LBDR18.x','LBDR26.x','LBDR31.x',
              'LBDR33.x','LBDR35.x','LBDR39.x','LBDR40.x','LBDR42.x','LBDR45.x','LBDR51.x','LBDR52.x','LBDR53.x',
              'LBDR54.x','LBDR55.x','LBDR56.x','LBDR58.x','LBDR59.x','LBDR61.x','LBDR62.x','LBDR64.x','LBDR66.x',
              'LBDR67.x','LBDR68.x','LBDR69.x','LBDR70.x','LBDR71.x','LBDR72.x','LBDR73.x','LBDR81.x','LBDR82.x',
              'LBDR83.x','LBDR84.x','LBDR89.x','LBDRPI.x','LBDRPCR.y','LBDRHP.y','LBDRLP.y','LBDR06.y','LBDR11.y',
              'LBDR16.y','LBDR18.y','LBDR26.y','LBDR31.y','LBDR33.y','LBDR35.y','LBDR39.y','LBDR40.y','LBDR42.y',
              'LBDR45.y','LBDR51.y','LBDR52.y','LBDR53.y','LBDR54.y','LBDR55.y','LBDR56.y','LBDR58.y','LBDR59.y',
              'LBDR61.y','LBDR62.y','LBDR64.y','LBDR66.y','LBDR67.y','LBDR68.y','LBDR69.y','LBDR70.y','LBDR71.y',
              'LBDR72.y','LBDR73.y','LBDR81.y','LBDR82.y','LBDR83.y','LBDR84.y','LBDR89.y','LBDRPI.y','WTSAF2YR.y',
              'PHAFSTHR.x','PHAFSTMN.x','WTSA2YR.x','PHAFSTHR.y','PHAFSTMN.y','WTSA2YR.y','WTSB2YR.y','WTSH2YR.y',
              'WTSB2YR.x','URXUCR.y','WTSOG2YR','WTSB2YR','LBXSCH','URDUMMAL','ACD011A','ACD011B','ACD011C','ACD040',
              'ACD110','ALQ101','ALQ110','ALQ120U','ALQ130','ALQ141Q',
              'ALQ141U','ALQ151','ALQ160','BPQ020','BPQ030','BPD035','BPQ040A','BPQ050A','BPQ056','BPD058',
              'BPQ059','BPQ080','BPQ060','BPQ070','BPQ090D','BPQ100D','CBD070','CBD090','CBD110','CBD120',
              'CBD130','HSD010','HSQ500','HSQ510','HSQ520','HSQ571','HSQ580','HSQ590','HSAQUEX','CSQ010',
              'CSQ020','CSQ030','CSQ040','CSQ060','CSQ070','CSQ080','CSQ090A','CSQ090B','CSQ090C','CSQ090D',
              'CSQ100','CSQ110','CSQ120A','CSQ120B','CSQ120C','CSQ120D','CSQ120E','CSQ120F','CSQ120G','CSQ120H',
              'CSQ140','CSQ160','CSQ170','CSQ180','CSQ190','CSQ200','CSQ202','CSQ204','CSQ210','CSQ220','CSQ240',
              'CSQ250','CSQ260','AUQ136','AUQ138','CDQ001','CDQ002','CDQ003','CDQ004','CDQ005','CDQ006','CDQ009A',
              'CDQ009B','CDQ009C','CDQ009D','CDQ009E','CDQ009F','CDQ009G','CDQ009H','CDQ008','CDQ010','DID040',
              'DIQ160','DIQ170','DIQ172','DIQ175A','DIQ175B','DIQ175C','DIQ175D','DIQ175E','DIQ175F','DIQ175G',
              'DIQ175H','DIQ175I','DIQ175J','DIQ175K','DIQ175L','DIQ175M','DIQ175N','DIQ175O','DIQ175P','DIQ175Q',
              'DIQ175R','DIQ175S','DIQ175T','DIQ175U','DIQ175V','DIQ175W','DIQ175X','DIQ180','DIQ050','DID060',
              'DIQ060U','DIQ070','DIQ230','DIQ240','DID250','DID260','DIQ260U','DIQ275','DIQ280','DIQ291','DIQ300S',
              'DIQ300D','DID310S','DID310D','DID320','DID330','DID341','DID350','DIQ350U','DIQ360','DIQ080',
              'DBQ010','DBD030','DBD041','DBD050','DBD055','DBD061','DBQ073A','DBQ073B','DBQ073C','DBQ073D',
              'DBQ073E','DBQ073U','DBQ700','DBQ197','DBQ223A','DBQ223B','DBQ223C','DBQ223D','DBQ223E','DBQ223U',
              'DBQ229','DBQ235A','DBQ235B','DBQ235C','DBQ301','DBQ330','DBQ360','DBQ370','DBD381','DBQ390','DBQ400',
              'DBD411','DBQ421','DBQ424','DBD895','DBD900','DBD905','DBD910','CBQ596','CBQ606','CBQ611','CBQ505',
              'CBQ535','CBQ540','CBQ545','CBQ550','CBQ552','CBQ580','CBQ585','CBQ590','DLQ010','DLQ020','DLQ040',
              'DLQ050','DLQ060','DLQ080','DED031','DEQ034A','DEQ034C','DEQ034D','DEQ038G','DEQ038Q','DED120',
              'DED125','DPQ010','DPQ020','DPQ030','DPQ040','DPQ050','DPQ060','DPQ070','DPQ080','DPQ090','DPQ100',
              'DUQ200','DUQ210','DUQ211','DUQ213','DUQ215Q','DUQ215U','DUQ219','DUQ220Q','DUQ220U','DUQ230',
              'DUQ240','DUQ250','DUQ260','DUQ270Q','DUQ270U','DUQ280','DUQ290','DUQ300','DUQ310Q','DUQ310U',
              'DUQ320','DUQ330','DUQ340','DUQ350Q','DUQ350U','DUQ352','DUQ360','DUQ370','DUQ380A','DUQ380B',
              'DUQ380C','DUQ380D','DUQ380E','DUQ390','DUQ400Q','DUQ400U','DUQ410','DUQ420','DUQ430','ECD010',
              'ECQ020','ECD070A','ECD070B','ECQ080','ECQ090','WHQ030E','MCQ080E','ECQ150','FSD032A','FSD032B',
              'FSD032C','FSD041','FSD052','FSD061','FSD071','FSD081','FSD092','FSD102','FSD032D','FSD032E',
              'FSD032F','FSD111','FSD122','FSD132','FSD141','FSD146','FSDHH','FSDAD','FSDCH','FSD151','FSQ165',
              'FSQ012','FSD012N','FSD230','FSD225','FSQ235','FSQ162','FSD650ZC','FSD660ZC','FSD675','FSD680',
              'FSD670ZC','FSQ690','FSQ695','FSD650ZW','FSD660ZW','FSD670ZW','HEQ010','HEQ020','HEQ030','HEQ040',
              'HIQ011','HIQ031A','HIQ031B','HIQ031C','HIQ031D','HIQ031E','HIQ031F','HIQ031G','HIQ031H','HIQ031I',
              'HIQ031J','HIQ031AA','HIQ260','HIQ105','HIQ270','HIQ210','HOD050','HOQ065','HUQ010','HUQ020','HUQ030',
              'HUQ041','HUQ051','HUQ061','HUQ071','HUD080','HUQ090','IMQ011','IMQ020','IMQ040','IMQ070','IMQ080',
              'IMQ090','IMQ045','INQ020','INQ012','INQ030','INQ060','INQ080','INQ090','INQ132','INQ140','INQ150',
              'IND235','INDFMMPI','INDFMMPC','INQ244','IND247','MCQ010','MCQ025','MCQ035','MCQ040','MCQ050','AGQ030',
              'MCQ053','MCQ070','MCQ075','MCQ080','MCQ082','MCQ084','MCQ086','MCQ092','MCD093','MCQ149','MCQ151',
              'MCQ160A','MCQ180A','MCQ195','MCQ160N','MCQ180N','MCQ160B','MCQ180B','MCQ160C','MCQ180C','MCQ160D',
              'MCQ180D','MCQ160E','MCQ180E','MCQ160F','MCQ180F','MCQ160G','MCQ180G','MCQ160M','MCQ170M','MCQ180M',
              'MCQ160K','MCQ170K','MCQ180K','MCQ160L','MCQ170L','MCQ180L','MCQ160O','MCQ203','MCQ206','MCQ220',
              'MCQ230A','MCQ230B','MCQ230C','MCQ230D','MCQ240A','MCQ240AA','MCQ240B','MCQ240BB','MCQ240C',
              'MCQ240CC','MCQ240D','MCQ240DD','MCQ240DK','MCQ240E','MCQ240F','MCQ240G','MCQ240H','MCQ240I',
              'MCQ240J','MCQ240K','MCQ240L','MCQ240M','MCQ240N','MCQ240O','MCQ240P','MCQ240Q','MCQ240R','MCQ240S',
              'MCQ240T','MCQ240U','MCQ240V','MCQ240W','MCQ240X','MCQ240Y','MCQ240Z','MCQ300A','MCQ300B','MCQ365A',
              'MCQ365B','MCQ365C','MCQ365D','MCQ370A','MCQ370B','MCQ370C','MCQ370D','MCQ380','OCD150','OCQ180',
              'OCQ210','OCQ260','OCD270','OCQ380','OCD390G','OCD395','OHQ030','OHQ033','OHQ770','OHQ780A','OHQ780B',
              'OHQ780C','OHQ780D','OHQ780E','OHQ780F','OHQ780G','OHQ780H','OHQ780I','OHQ780J','OHQ780K','OHQ555G',
              'OHQ555Q','OHQ555U','OHQ560G','OHQ560Q','OHQ560U','OHQ565','OHQ570Q','OHQ570U','OHQ575G','OHQ575Q',
              'OHQ575U','OHQ580','OHQ585Q','OHQ585U','OHQ590G','OHQ590Q','OHQ590U','OHQ610','OHQ612','OHQ614',
              'OHQ620','OHQ640','OHQ680','OHQ835','OHQ845','OHQ848G','OHQ848Q','OHQ849','OHQ850','OHQ855','OHQ860',
              'OHQ865','OHQ870','OHQ875','OHQ880','OHQ885','OHQ895','OHQ900','OSQ010A','OSQ010B','OSQ010C',
              'OSQ020A','OSQ020B','OSQ020C','OSD030AA','OSQ040AA','OSD050AA','OSD030AB','OSQ040AB','OSD050AB',
              'OSD030AC','OSQ040AC','OSD050AC','OSD030BA','OSQ040BA','OSD050BA','OSD030BB','OSQ040BB','OSD050BB',
              'OSD030BC','OSQ040BC','OSD050BC','OSD030BD','OSQ040BD','OSD050BD','OSD030BE','OSQ040BE','OSD030BF',
              'OSQ040BF','OSD030BG','OSQ040BG','OSD030BH','OSQ040BH','OSD030BI','OSQ040BI','OSD030BJ','OSQ040BJ',
              'OSD030CA','OSQ040CA','OSD050CA','OSD030CB','OSQ040CB','OSD050CB','OSD030CC','OSQ040CC','OSQ080',
              'OSQ090A','OSQ100A','OSD110A','OSQ120A','OSQ090B','OSQ100B','OSD110B','OSQ120B','OSQ090C','OSQ100C',
              'OSD110C','OSQ120C','OSQ090D','OSQ100D','OSD110D','OSQ120D','OSQ090E','OSQ100E','OSD110E','OSQ120E',
              'OSQ090F','OSQ120F','OSQ090G','OSQ100G','OSD110G','OSQ120G','OSQ090H','OSQ120H','OSQ060','OSQ072',
              'OSQ130','OSQ140Q','OSQ140U','OSQ150','OSQ160A','OSQ160B','OSQ170','OSQ180','OSQ190','OSQ200',
              'OSQ210','OSQ220','PFQ020','PFQ030','PFQ033','PFQ041','PFQ049','PFQ051','PFQ054','PFQ057','PFQ059',
              'PFQ061A','PFQ061B','PFQ061C','PFQ061D','PFQ061E','PFQ061F','PFQ061G','PFQ061H','PFQ061I','PFQ061J',
              'PFQ061K','PFQ061L','PFQ061M','PFQ061N','PFQ061O','PFQ061P','PFQ061Q','PFQ061R','PFQ061S','PFQ061T',
              'PFQ063A','PFQ063B','PFQ063C','PFQ063D','PFQ063E','PFQ090','PAQ605','PAQ610','PAD615','PAQ620',
              'PAQ625','PAD630','PAQ635','PAQ640','PAD645','PAQ650','PAQ655','PAD660','PAQ665','PAQ670','PAD675',
              'PAD680','PAQ706','PAQ710','PAQ715','PAQ722','PAQ724A','PAQ724B','PAQ724C','PAQ724D','PAQ724E',
              'PAQ724F','PAQ724G','PAQ724H','PAQ724I','PAQ724J','PAQ724K','PAQ724L','PAQ724M','PAQ724N','PAQ724O',
              'PAQ724P','PAQ724Q','PAQ724R','PAQ724S','PAQ724T','PAQ724U','PAQ724V','PAQ724W','PAQ724X','PAQ724Y',
              'PAQ724Z','PAQ724AA','PAQ724AB','PAQ724AC','PAQ724AD','PAQ724AE','PAQ724AF','PAQ724CM','PAQ731',
              'PAD733','PAQ677','PAQ678','PAQ740','PAQ742','PAQ744','PAQ746','PAQ748','PAQ755','PAQ759A',
              'PAQ759B','PAQ759C','PAQ759D','PAQ759E','PAQ759F','PAQ759G','PAQ759H','PAQ759I','PAQ759J','PAQ759K',
              'PAQ759L','PAQ759M','PAQ759N','PAQ759O','PAQ759P','PAQ759Q','PAQ759R','PAQ759S','PAQ759T','PAQ759U',
              'PAQ759V','PAQ762','PAQ764','PAQ766','PAQ679','PAQ750','PAQ770','PAQ772A','PAQ772B','PAQ772C',
              'PAAQUEX','PUQ100','PUQ110','RHQ010','RHQ020','RHQ031','RHD043','RHQ060','RHQ070','RHQ074','RHQ076',
              'RHQ078','RHQ131','RHD143','RHQ160','RHQ162','RHQ163','RHQ166','RHQ169','RHQ172','RHD173','RHQ171',
              'RHD180','RHD190','RHQ197','RHQ200','RHD280','RHQ291','RHQ305','RHQ332','RHQ420','RHQ540','RHQ542A',
              'RHQ542B','RHQ542C','RHQ542D','RHQ554','RHQ560Q','RHQ560U','RHQ570','RHQ576Q','RHQ576U','RHQ580',
              'RHQ586Q','RHQ586U','RHQ596','RHQ602Q','RHQ602U','RXQ510','RXQ515','RXQ520','RXQ525G','RXQ525Q',
              'RXQ525U','RXD530','SLD010H','SLQ050','SLQ060','SMQ020','SMD030','SMQ040','SMQ050Q','SMQ050U',
              'SMD055','SMD057','SMQ078','SMD641','SMD093','SMDUPCA','SMD100BR','SMD100FL','SMD100MN','SMD100LN',
              'SMD100TR','SMD100NI','SMD100CO','SMQ621','SMD630','SMQ661','SMQ665A','SMQ665B','SMQ665C','SMQ665D',
              'SMQ670','SMQ848','SMQ852Q','SMQ852U','SMAQUEX2','SMD460','SMD470','SMD480','SMQ856','SMQ858',
              'SMQ860','SMQ862','SMQ866','SMQ868','SMQ870','SMQ872','SMQ874','SMQ876','SMQ878','SMQ880',
              'SMAQUEX.x','SMQ681','SMQ690A','SMQ710','SMQ720','SMQ725','SMQ690B','SMQ740','SMQ690C','SMQ770',
              'SMQ690G','SMQ845','SMQ690H','SMQ849','SMQ851','SMQ690D','SMQ800','SMQ690E','SMQ817','SMQ690I',
              'SMQ857','SMQ690J','SMQ861','SMQ863','SMQ690F','SMQ830','SMQ840','SMDANY','SMAQUEX.y','SXD021',
              'SXQ800','SXQ803','SXQ806','SXQ809','SXQ700','SXQ703','SXQ706','SXQ709','SXD031','SXD171','SXD510',
              'SXQ824','SXQ827','SXD633','SXQ636','SXQ639','SXD642','SXQ410','SXQ550','SXQ836','SXQ841','SXQ853',
              'SXD621','SXQ624','SXQ627','SXD630','SXQ645','SXQ648','SXQ590','SXQ600','SXD101','SXD450','SXQ724',
              'SXQ727','SXQ130','SXQ490','SXQ741','SXQ753','SXQ260','SXQ265','SXQ267','SXQ270','SXQ272','SXQ280',
              'SXQ292','SXQ294','WHD010','WHD020','WHQ030','WHQ040','WHD050','WHQ060','WHQ070','WHD080A','WHD080B',
              'WHD080C','WHD080D','WHD080E','WHD080F','WHD080G','WHD080H','WHD080I','WHD080J','WHD080K','WHD080M',
              'WHD080N','WHD080O','WHD080P','WHD080Q','WHD080R','WHD080S','WHD080T','WHD080U','WHD080L','WHD110',
              'WHD120','WHD130','WHD140','WHQ150','WHQ030M','WHQ500','WHQ520'],axis=1)

df.shape #(9813, 48)

df = df.rename(columns = {'SEQN' : 'ID', 'RIAGENDR' : 'Gender','RIDAGEYR': 'Age','DMDYRSUS' : 'Years_in_US',
                        'RIDRETH3': 'Race', 'DMDBORN4': 'Country_of_Birth','DMDCITZN':'Citizenship',
                        'DMDMARTL':'Marital_Status', 'RIDEXPRG': 'Pregnancy_Status','SIALANG':'Language',
                        'DMDFMSIZ':'Family_Size','INDFMIN2':'Family_income','BPXSY1': 'Systolic_BP1',
                        'BPXDI1': 'Diastolic_BP1','BPXSY2':'Systolic_BP2','BPXDI2': 'Diastolic_BP2',
                        'BMXWT': "Weight",'BMXHT': 'Height', 'BMXBMI': 'BodyMassIndex','MGXH1T1':'GripStrength_left',
                        'MGXH2T1': 'GripStrength_right','URXUMS':'Albumin','URXUCR.x':'Creatinine',
                        'LBDAPBSI':'Apoliprotein','LBXSGB':'Globulin','LBXSGL':'Glucose','LBXSTP':'Total_Protein',
                        'LBXWBCSI':'White_blood_cells','LBXRBCSI':'Red_blood_cells','LBXHGB':'Hemoglobin',
                        'LBXPLTSI':'Blood_platelets','LBXHA':'HepatitisA_antibody','LBXIN':'Insulin',
                        'LBXTC':'Cholesterol','LBXBPB':'Blood_lead','LBXBCD':'Blood_cadmium',
                        'LBXTHG':'Blood_mercury','LBXBSE':'Blood_selenium','LBXBMN':'Blood_manganese',
                        'URXPREG':'Pregnancy_test','LBDB12':'Vitamin_B12','ALQ120Q':'AlcoholConsumption_yearly',
                        'DIQ010':'Diabetes','DUQ217':'MarijuanaUse_monthly','DUQ272':'CocaineConsumption',
                        'MCQ300C':'Family_Diabetes','SMD650':'SmokingFrequency_monthly',
                          'SXQ610':'SexualActivity_yearly','SXQ251':'Unprotected_Sex'})

df= df[df.Family_income != 13.0]
df= df[df.Family_income != 12.0]
df= df[df.Family_income != 11.0]
df= df[df.Family_income != 77.0]
df= df[df.Family_income != 99.0]
df['Family_income'] = df['Family_income'].fillna(df['Family_income'].median())
df.shape #(9186, 48)

df= df[df.Country_of_Birth != 77]
df= df[df.Country_of_Birth != 99]
df.shape #(9184, 48)

df= df[df.Citizenship != 7]
df= df[df.Citizenship != 9]
df['Citizenship'] = df['Citizenship'].fillna(df['Citizenship'].median())
df.shape #(9177, 48)

df= df[df.Marital_Status != 77]
df= df[df.Marital_Status != 99]
df['Marital_Status'] = df['Marital_Status'].fillna('5')
df.shape #(9175, 48)

df = df[df.Age != 0]
df['Pregnancy_Status'] = df['Pregnancy_Status'].fillna('2')
df.shape #(8812, 48)

df = df[pd.notnull(df['Systolic_BP1'])]
df = df[pd.notnull(df['Diastolic_BP1'])]
df = df[pd.notnull(df['Systolic_BP2'])]
df = df[pd.notnull(df['Diastolic_BP2'])]
df.shape #(6628, 48)

df = df[pd.notnull(df['Weight'])]
df = df[pd.notnull(df['Height'])]
df = df[pd.notnull(df['BodyMassIndex'])]
df.shape #(6568, 48)

df = df[pd.notnull(df['GripStrength_left'])]
df = df[pd.notnull(df['GripStrength_right'])]
df.shape #(6170, 48)

df = df[pd.notnull(df['Albumin'])]
df.shape #(6059, 48)

df = df[pd.notnull(df['Creatinine'])]
df = df[pd.notnull(df['Globulin'])]
df = df[pd.notnull(df['Glucose'])]
df = df[pd.notnull(df['Total_Protein'])]
df = df[pd.notnull(df['White_blood_cells'])]
df = df[pd.notnull(df['Red_blood_cells'])]
df = df[pd.notnull(df['Hemoglobin'])]
df = df[pd.notnull(df['Blood_platelets'])]
df = df[pd.notnull(df['HepatitisA_antibody'])]
df = df[pd.notnull(df['Cholesterol'])]
df = df[pd.notnull(df['Cholesterol'])]
df.shape #(5101, 48)

df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
df['Blood_lead'] = df['Blood_lead'].fillna(df['Blood_lead'].median())
df['Blood_cadmium'] = df['Blood_cadmium'].fillna(df['Blood_cadmium'].median())
df['Blood_mercury'] = df['Blood_mercury'].fillna(df['Blood_mercury'].median())
df['Blood_selenium'] = df['Blood_selenium'].fillna(df['Blood_selenium'].median())
df['Blood_manganese'] = df['Blood_manganese'].fillna(df['Blood_manganese'].median())

df['Pregnancy_test'] = df['Pregnancy_test'].fillna('2')
df['Vitamin_B12'] = df['Vitamin_B12'].fillna(df['Vitamin_B12'].median())
df.shape #(5101, 48)

df['Unprotected_Sex'] = df['Unprotected_Sex'].fillna('1')
df = df[df.Unprotected_Sex != 7]
df = df[df.Unprotected_Sex != 9]
df['SexualActivity_yearly'] = df['SexualActivity_yearly'].fillna('1')
df = df[df.SexualActivity_yearly != 77]
df = df[df.SexualActivity_yearly != 99]
df['AlcoholConsumption_yearly'] = df['AlcoholConsumption_yearly'].fillna('0')
df = df[df.AlcoholConsumption_yearly != 999]
df = df[df.AlcoholConsumption_yearly != 777]
df['Diabetes'] = df['Diabetes'].fillna('2')
df = df[df.Diabetes != 7]
df = df[df.Diabetes != 9]
df['MarijuanaUse_monthly'] = df['MarijuanaUse_monthly'].fillna('0')
df = df[df.MarijuanaUse_monthly != 9]
df = df[df.MarijuanaUse_monthly != 7]
df['CocaineConsumption'] = df['CocaineConsumption'].fillna('0')
df = df[df.CocaineConsumption != 9]
df = df[df.CocaineConsumption != 7]
df['Family_Diabetes'] = df['Family_Diabetes'].fillna('2')
df = df[df.Family_Diabetes != 7]
df = df[df.Family_Diabetes != 9]
df['SmokingFrequency_monthly'] = df['SmokingFrequency_monthly'].fillna('0')
df = df[df.SmokingFrequency_monthly != 999]
df = df[df.SmokingFrequency_monthly != 777]
df.shape #(5011, 48)
