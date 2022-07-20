import os
import sys
import logging
import optparse
import xml.etree.ElementTree as ET


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

ROOT_PATH = os.path.join(os.environ['HOME'], 'VisualSearch')
COLLECTION = 'iacc.3'
EDITION = 'tv18'


def parse_result(res):
    resp = {}
    lines = res.split('\n')
    for line in lines:
        elems = line.split()
        if 'infAP' == elems[0] and 'all' in line:
            return float(elems[-1])


def xml_to_treceval(opt, input_file):
    overwrite = opt.overwrite

    res_file = os.path.splitext(input_file)[0] + '.treceval'

    if os.path.exists(res_file):
        if overwrite:
            logger.info('%s exists. Overwrite' % res_file)
        else:
            logger.info('%s exists. Use "--overwrite 1" if you want to overwrite' % res_file)
            return res_file

    tree = ET.parse(input_file)
    root = tree.getroot()

    MAX_SCORE = 9999
    TEAM = 'RUCMM'

    newlines = []
    for topicResult in root.iter('videoAdhocSearchTopicResult'):
        qry_id = '1' + topicResult.attrib['tNum']
        itemlist = list(topicResult)
        for rank, item in enumerate(itemlist):
            assert(rank+1 == int(item.attrib['seqNum']))
            shot_id = item.attrib['shotId']
            score = MAX_SCORE - rank
            newlines.append('%s 0 %s %d %d %s' % (qry_id, shot_id, rank+1, score, TEAM))

    fw = open(res_file, 'w')
    fw.write('\n'.join(newlines)+'\n')
    fw.close()

    return res_file


def process(opt, input_xml_file):

    treceval_file = xml_to_treceval(opt, input_xml_file)
    # res_file = os.path.join(os.path.dirname(input_xml_file), 'perf.txt')
    res_file = input_xml_file+'_perf.txt'
    gt_file = os.path.join(opt.rootpath, opt.collection, 'TextData', 'avs.qrels.%s' % opt.edition)

    os.chdir(os.path.abspath(os.path.dirname(__file__)))  # 改变工作目录
    cmd = 'perl sample_eval.pl -q %s %s' % (gt_file, treceval_file)
    res = os.popen(cmd).read()

    with open(res_file, 'w') as fw:
        fw.write(res)

    resp = parse_result(res)

    print('%s infAP: \t%.3f' % (opt.edition, resp), end="\t")

    return resp


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser
    parser = OptionParser(usage="""usage: %prog [options] input_xml_file""")
    parser.add_option('--rootpath', type=str, default=ROOT_PATH, help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_option('--collection', type=str, default=COLLECTION, help='test collection')
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default: 0)")
    parser.add_option("--edition", default=EDITION, type="string", help="trecvid edition (default: %s)" % EDITION)
            
    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1

    if '~' in options.rootpath:
        options.rootpath = options.rootpath.replace('~', os.path.expanduser('~'))
    return process(options, args[0])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv = "txt2xml.py " \
                   "/data2/hf/VisualSearch/v3c1/SimilarityIndex/Sea_plus_multi_head_/TV20/runs_pretrain_gcc_old_mean_last_11_12_1_12_seed_2/id.sent.score.txt.xml " \
                   "--collection v3c1 " \
                   "--rootpath ~/hf_code/VisualSearch --edition tv20 " \
                   "--overwrite 1".split(' ')
        sys.exit(main())
    sys.exit(main())
