import glob
import json
#import scrubadub
import re

SOURCE_PATH = '/Users/iseletsk/lve/randomHelpers/data/2023-01-06/json'

AUTHORS = {}


def init_all_users():
    for file in glob.glob(f'{SOURCE_PATH}/users.*.json'):
        with open(file, 'r') as f:
            users = json.load(f)
            for user in users['users']:
                AUTHORS[user['id']] = user


def end_user(user_id):
    return AUTHORS[user_id]['role'] == 'end-user'


ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
url_pattern = r"(?P<url>https?://[^\s]+)"
email_pattern = r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
phone_pattern = r"(\+?\d[\d -]{8,12}\d)"
#hostname_patterns = r"([a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
homedir_pattern = r"(/home/[^/]+/[^/]+)"


def filter_token(token, skiplist=[]):
    for entry in skiplist:
        if entry in token:
            return False
    return True


def filter_tokens(tokens, skiplist=[]):
    return [token for token in tokens if filter_token(token, skiplist)]


def replace_token(all_tokens, text, pattern, token_name, skiplist=[]):
    tokens = filter_tokens(re.findall(pattern, text), skiplist)

    for token in tokens:
        if token not in all_tokens:
            all_tokens.append(token)
    for i, token in enumerate(all_tokens):
        text = text.replace(token, f'[{token_name}{i}]')
    return text


def clean_tokens(ticket_cache, text):
    text = replace_token(ticket_cache['ips'], text, ip_pattern, 'ip')
    text = replace_token(ticket_cache['urls'], text, url_pattern, 'url', ['cloudlinux.com', 'cloudlinux.zendesk.com', 'imunify.com', 'imunify360.com', 'cpanel.net', 'plesk.com'])
    text = replace_token(ticket_cache['emails'], text, email_pattern, 'email', ['cloudlinux.com', 'tuxcare.com'])
#    text = replace_token(ticket_cache['phones'], text, phone_pattern, 'phone')
#    text = replace_token(ticket_cache['hostnames'], text, hostname_patterns, 'hostname')
    text = replace_token(ticket_cache['homedirs'], text, homedir_pattern, 'homedir')
    return text


def init_ticket_cache():
    return {
        'ips': [],
        'urls': [],
        'emails': [],
        'phones': [],
 #       'hostnames': [],
        'homedirs': []
    }


def get_all_ticket_ids():
    ids = []
    for file in glob.glob(f'{SOURCE_PATH}/ticket.*.json'):
        ids.append(int(file.split('.')[-2]))
    return ids


def ticket_parse(ticket_id):
    with open(f'{SOURCE_PATH}/ticket.{ticket_id}.json', 'r') as f:
        ticket = json.load(f)['ticket']
        subject = ticket['subject']
        tags = ticket['tags']

    comments = []
    with open(f'{SOURCE_PATH}/comment.{ticket_id}.json', 'r') as f:
        comments_json = json.load(f)
        for comment in comments_json['comments']:
            if comment["public"] is False:
                continue
            body = comment['body']
            author = AUTHORS[comment['author_id']]
            is_end_user = False
            if author['role'] == 'end-user':
                is_end_user = True
            elif (author['name'].lower()) in ('cloudlinux zendesk', 'cloudlinux support'):
                continue

            if not is_end_user:
                if 'Best Regards,' in body:
                    body = body[:body.index('Best Regards')]
                if 'Regards,' in body:
                    body = body[:body.index('Regards,')]
            comments.append({
                'end-user': is_end_user,
                'body': body, })
    return {
        "subject": subject,
        'comments': comments,
        'tags': tags,
        'status': ticket['status'],
    }


def clean_comment(ticket_cache, comment):
    comment = comment.replace('\n\n', '\n').replace('* * *', '')
    comment = clean_tokens(ticket_cache, comment)
    return comment #scrubadub.clean(comment)


def ticket_to_jsonl(ticket_id):
    ticket = ticket_parse(ticket_id)
    ticket_cache = init_ticket_cache()
    prompt = f'Summary: {ticket["subject"]}\n\n###\n\n'
    comments = ticket['comments']
    try:
        last_comment = comments.pop()
        while last_comment['end-user']:
            last_comment = comments.pop()
    except IndexError:
        return None

    for comment in comments:
        body = clean_comment(ticket_cache, comment['body'])
        if comment['end-user']:
            prompt += f'Customer: {body}\n\n'
        else:
            prompt += f'Agent: {body}\n\n'

    completion = f' {clean_comment(ticket_cache, last_comment["body"])}\n'
    return {
        'id': ticket_id,
        'prompt': prompt,
        'completion': completion,
        'tags': ticket['tags'],
        'status': ticket['status'],
    }


init_all_users()
with open('workdir/tickets_list.jsonl', 'w') as f:
    for ticket_id in get_all_ticket_ids()[:10]:
        result = ticket_to_jsonl(ticket_id)
        if result is not None:
            f.write(json.dumps(result))
            f.write('\n')
#print(ticket_to_jsonl(167927))
#print(ticket_to_jsonl(163502))

