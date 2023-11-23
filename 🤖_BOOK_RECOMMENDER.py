import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import plotly.express as px


#SET PAGE CONFIGURATION
st.set_page_config(
    page_title="BOOK RECOMMENDER",
    page_icon="📚",
    layout="wide"
)


#APPLY CSS
with open('pages.css',
          'r', encoding = 'utf-8') as f:
    st.markdown(f'<style>{f.read()}<style>', unsafe_allow_html = True)

#CACHE DATA
@st.cache_data(persist = True)
def getdata():
    books_df = pd.read_csv('data/full_info_df.csv')
    similarity_df = pd.read_csv('data/similarity_df.csv', index_col = 'Unnamed: 0')
    return books_df, similarity_df
books_df, similarity_df = getdata()[0], getdata()[1]

#BOOK CHOSING ALGORITMS
def choose_similar_book(book_name): 
    try:
        cluster = int(books_df.loc[books_df['book_name'] == book_name]['clusters'])
    except:
        cluster = int(books_df.loc[books_df['book_name'] == book_name]['clusters'].iloc[0])
    cluster_matches = books_df.loc[books_df['clusters'] == cluster]['book_name'].tolist()
    book_matches = list(similarity_df.loc[cluster_matches][book_name].sort_values(ascending = True)[1:6].index)
    results = books_df.loc[books_df['book_name'].isin(book_matches),['book_name','price', 'img', 'describe', 'detail_cate', 'large_cate', 'cover_type', 'num_of_pages', 'rating star', 'book_length', 'book_width']]

    return results


#MAIN MENU

menu1, menu2 = st.columns([1,4])

with menu1:
    st.image('https://scontent.xx.fbcdn.net/v/t1.15752-9/370289889_1084884782525490_5622143318006968801_n.png?stp=dst-png_p206x206&_nc_cat=104&ccb=1-7&_nc_sid=510075&_nc_ohc=UXl8fV7ARHkAX_Q7irN&_nc_ad=z-m&_nc_cid=0&_nc_ht=scontent.xx&oh=03_AdQN8u-39ShIBB_hfUo0aJiOTBtzLufp14PONKbXIvgFuA&oe=658644D3',
             width = 200)
with menu2:
    options = option_menu(
        menu_title = 'Welcome to Book Recommender, please choose what you want',
        options = ['BOOK RECOMMENDER', 'BOOK MARKET', 'HOW THIS APP WORKS?'],
        icons = ['robot','book','wrench'],
        menu_icon = 'window-dock',
        orientation = 'horizontal',
        styles = {
            'container': {'background-image': 'url("https://images.pexels.com/photos/1323550/pexels-photo-1323550.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2")',
                          'background-size': 'cover'},
            'menu-title': {'font-size': '20','font-family': 'cursive', 'text-weight': 'bold'},
            'nav-link': {'font-size': '15', "text-align": "left", "margin":"0px", "--hover-color": "#eee", "font-family": "cursive"},
            'nav-link-selected': {'background-color': 'lightblue'}
        }
    )

#EXPLAINATIONS:
if options == 'HOW THIS APP WORKS?':
    for i in range(0,5):
        st.markdown('')
    st.markdown('# How this app work?')
    st.divider()
    st.markdown('The recommendation system app employs based on a series of algorithms'
                ' of unsupervised machine learning and deep learning techniques')
    st.markdown('STEP1: WEB SCARPLING')
    st.text('')
    st.markdown('- The idea is access to Tiki and scarpling book info in: https://tiki.vn/sach-truyen-tieng-viet/')
    st.markdown('- Sample Data after processing:')
    sampledata = books_df.drop(['Unnamed: 0', 'wv_describe', 'clusters'], axis = 1)
    st.write(sampledata.head())
    st.markdown('- Tool to use: Selenium with Threading')
    st.markdown('The text scarpling look like this:')
    with st.expander('Click to see scarpling code'):
        st.code('''
def open_multi_browsers(n_page):
    drivers = []
    for _ in range(n_page):
        driver = webdriver.Chrome()
        drivers.append(driver)
    return drivers

def load_multi_pages(driver, n):
    driver.maximize_window()
    link = f'https://tiki.vn/sach-truyen-tieng-viet/c316?page={n}'
    driver.get(link)
    sleep(3)

def load_multi_browsers(drivers, idx_page):
    for driver, page in zip(drivers, idx_page):
        t = threading.Thread(target = load_multi_pages, args = (driver, page))
        t.start()

def get_data(driver, que):
    try:
        prod_links_elems = driver.find_elements(By.CSS_SELECTOR, '.style__ProductLink-sc-7xd6qw-2.fHwskZ.product-item')
        prod_links = [i.get_attribute('href') for i in prod_links_elems]
    except TimeoutException:
        wait = WebDriverWait(driver, 10)
        element_to_wait = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.style__ProductLink-sc-7xd6qw-2.fHwskZ.product-item')))
        prod_links_elems = driver.find_elements(By.CSS_SELECTOR, '.style__ProductLink-sc-7xd6qw-2.fHwskZ.product-item')
        prod_links = [i.get_attribute('href') for i in prod_links_elems]

    page_prod_features = []

    for prod_link in prod_links:
        driver.get(prod_link)
        sleep(2)
        driver.maximize_window()
        scroll_iterations = 10
        scroll_amount = 300
        scroll_interval = 0.2 

        for _ in range(scroll_iterations):
            driver.execute_script("window.scrollBy(0, arguments[0]);", scroll_amount)
            sleep(scroll_interval)

        try:
            wait = WebDriverWait(driver, 10)
            element_to_wait = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '.btn-more')))
            element_to_wait.click()
        except TimeoutException:
            print('Not btn-more')

        try:
            category_elems = driver.find_elements(By.CSS_SELECTOR, '.Breadcrumb__Wrapper-sc-1r2fjia-0.gsoENx .breadcrumb-item')
            category = [i.text for i in category_elems]
        except NoSuchElementException:
            category = np.nan

        try:
            img_elem = driver.find_element(By.CSS_SELECTOR, '.image-frame')
            img = img_elem.find_element(By.TAG_NAME, 'img').get_attribute('srcset').split(' ')[0]
        except NoSuchElementException:
            img = np.nan

        try:
            price = driver.find_element(By.CSS_SELECTOR, '.product-price__current-price').text
        except NoSuchElementException:
            price = np.nan

        try:
            discount = driver.find_element(By.CSS_SELECTOR, '.product-price__discount-rate').text
        except NoSuchElementException:
            discount = np.nan

        try:
            sale_quantities = driver.find_element(By.CSS_SELECTOR, '.styles__StyledQuantitySold-sc-1swui9f-3.bExXAB').text
        except NoSuchElementException:
            sale_quantities = np.nan

        try:
            rating = driver.find_element(By.CSS_SELECTOR, '.styles__StyledReview-sc-1swui9f-1.dXPbue').text
        except NoSuchElementException:
            rating = np.nan

        

        info_elems = driver.find_elements(By.CSS_SELECTOR, '.WidgetTitle__WidgetContainerStyled-sc-1ikmn8z-0.iHMNqO')
        for i in info_elems:
            try:
                title = i.find_element(By.CSS_SELECTOR, '.WidgetTitle__WidgetTitleStyled-sc-1ikmn8z-1.eaKcuo').text
                print(title)
                if title == 'Thông tin chi tiết':
                    info_row = i.find_elements(By.CSS_SELECTOR, '.WidgetTitle__WidgetContentStyled-sc-1ikmn8z-2.jMQTPW')
                    info = [i.text.split('/n') for i in info_row]
                    print('Success collect info')
                elif title == 'Mô tả sản phẩm':
                    describe = i.find_element(By.CSS_SELECTOR, '.style__Wrapper-sc-13sel60-0.dGqjau.content').text
                    print('Success collect describe')
                elif title == 'Thông tin nhà bán':
                    seller = i.find_element(By.CSS_SELECTOR, '.seller-name').text.split(' ')[0]
                    seller_evaluation_elems = i.find_element(By.CSS_SELECTOR, '.item.review')
                    seller_star = seller_evaluation_elems.find_element(By.CSS_SELECTOR, '.title').text
                    seller_reviews_quantity = seller_evaluation_elems.find_element(By.CSS_SELECTOR, '.sub-title').text
                    seller_follow = i.find_element(By.CSS_SELECTOR, '.item.normal .title').text
                    print('Succes collect seller info')
            except NoSuchElementException:
                print('PASS')

        features = [category, img, price, discount, sale_quantities, rating, info, describe, seller, seller_star, seller_reviews_quantity, seller_follow]
        page_prod_features.append(features)
        
    que.put(page_prod_features)

def runInParallbel(func, drivers):
    threads = []
    que = Queue()

    for driver in drivers:
        print('--Running--')
        t = threading.Thread(target = func, args = (driver, que))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    
    results = []
    while not que.empty():
        results.extend(que.get())
    
    return results
    n_page = 5
    drivers = open_multi_browsers(n_page)
    idx_page = [i for i in range(1, n_page + 1)]
    all_data = pd.DataFrame()
    while idx_page[0] < 50:
        load_multi_browsers(drivers, idx_page)
        sleep(5)
        all_prod_features = runInParallbel(get_data, drivers)
        page_df = pd.DataFrame(all_prod_features, columns = ['category', 'img', 'price', 'discount', 'sale_quantities', 'rating', 'info', 'describe', 'seller', 'seller_star', 'seller_reviews_quantity', 'seller_follow'])
        all_data = pd.concat([all_data, page_df], axis = 0)
        idx_page = [i + 5 for i in idx_page]
    ''', language = 'python')
    st.markdown('STEP2: TEXT PROCESSING')
    st.text('')
    st.markdown('- In this I use Natural Language Processing to transform text data')
    st.markdown('- Tool to use: Spacy for tokenizing and Gensim FastText for text vectorizing')
    st.markdown('1. Tokenizing - Use Spacy Vietnamese pre-train model: vi_core_news_lg')
    st.markdown('2.  Text vectorizing - Use FastText model build for Vietnamese: cc.vi.300.bin')
    with st.expander('Click to see text processing code:'):
        st.code('''
model_path = 'C:/Users/admin/Documents/Data Science/nlp/vnmodel/cc.vi.300.bin'
nlp = spacy.load('vi_core_news_lg')
fasttext_model = FastText.load_fasttext_format(model_path)
                
def preprocessing_vectorizing(text):
    filtered_token = []
    for i in nlp(text.lower()):
        if not i.is_punct and not i.is_stop:
            filtered_token.append(i)
    v = [fasttext_model.wv[token.text] for token in filtered_token]
    return np.mean(v, axis = 0)
                
df['wv_describe'] = df['describe'].map(preprocessing_vectorizing)
''', language = 'python')
    st.markdown('STEP3: CHOOSING BOOK ALGORITHMS')
    st.text('')
    st.markdown('- For choosing the best relative books, I make 2'
                ' steps:\n1. KMeans Clustering for divide similar book in a clusters.'
                ' \n2. Cosine Similarity the describes that be vectorized before to choose sort same level of books')
    st.markdown('- Tool to use: KMeans Clustering and Cosine similarity')
    with st.expander('Click to see choosing book algorithms code'):
        st.code('''
def classify_cols(X):
    num_col = list(X.select_dtypes(['float64','int64','int32']).columns)
    highcar_cat_col = [i for i in X.columns if i not in num_col and X[i].nunique() > 10]
    lowcar_cat_col = [i for i in X.columns if i not in num_col and X[i].nunique() <= 10]
    return num_col, highcar_cat_col, lowcar_cat_col
                
wv_matrix = np.stack(np.array(df['wv_describe']))
distance = 1 - cosine_similarity(wv_matrix)
vals = df['book_name'].tolist()
sl_df = pd.DataFrame(distance, columns = vals, index = vals)
sl_df = sl_df.round(3)
                
X = df.loc[:,['price', 'detail_cate', 'large_cate']]
y = df['sale_quantities']
                
num_col, highcar_cat_col, lowcar_cat_col = classify_cols(X)
                
num_tfmer = Pipeline(steps = [
    ('impute', SimpleImputer(strategy = 'median')),
    ('scaling', StandardScaler())
])

lowcar_tfmer = Pipeline(steps = [
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('encode', OneHotEncoder(sparse_output = False, handle_unknown = 'ignore'))
])

highcar_tfmer = Pipeline(steps = [
    ('impute', SimpleImputer(strategy = 'most_frequent')),
    ('encode', MEstimateEncoder()),
    ('scale', StandardScaler())
])

preprocessor = ColumnTransformer(transformers = [
    ('num', num_tfmer, num_col),
    ('high', highcar_tfmer, highcar_cat_col),
    ('low', lowcar_tfmer, lowcar_cat_col)
])
                
X_pp = preprocessor.fit_transform(X, y)

#Plot to choose best k_clusters             
sse = []
k_range = range(1,10)
for i in k_range:
    km = KMeans(n_clusters = i)
    km.fit_predict(X_pp)
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('sse')
plt.plot(k_range, sse)
                
km = KMeans(n_clusters = 7)
df['clusters'] = km.fit_predict(X_pp)
                
def choose_similar_book(book_name):
    cluster = int(df.loc[df['book_name'] == book_name]['clusters'])
    cluster_matches = df.loc[df['clusters'] == cluster]['book_name'].tolist()
    cosine_matches = list(sl_df.loc[cluster_matches][book_name].sort_values(ascending = True)[:5].index)
    return cosine_matches
''', language = 'python')
    st.markdown('That is what I do for recommending book. Hope you enjoy it.')
    for i in range(0,10):
        st.markdown('')



#RECOMMENDATIONS
if options == 'BOOK RECOMMENDER':
    st.header('SELECT A BOOK TO SEE THE RECOMMENDATION')

    selected_book = st.selectbox(
        'Please select your book',
        options = [''] + books_df['book_name'].to_list(),
        key = 'button',
        format_func = lambda x: 'Select a book' if x == '' else x,
        index = 0
    )

    def convert(x, type):
        if type == 0:
            try:
                x = str(round(x))
            except:
                x = 'Chưa có thông tin'
        elif type == 1:
            try:
                x = str(round(x,1))
            except:
                x = 'Chưa có thông tin'
        elif type == 2:
            try:
                x = str(round(x)) + 'cm'
            except:
                x = 'Chưa có thông tin'
        return x

    if st.button('Show recommendation'):
        info = books_df[books_df['book_name'] == selected_book]
        st.markdown(f'## I. Chi tiết về sách bạn chọn:')
        st.markdown(f'### Tiêu đề: {selected_book}')         
        price = '{:,}'.format(round(info['price'].values[0]))
        st.markdown(f'Giá: {price} vnd')

        st.image(str(info['img'].values[0]))
        with st.expander('Mô tả về sách bạn chọn'):
            st.write(str(info['describe'].values[0]))
        book_info = pd.DataFrame(info[['detail_cate', 'large_cate', 'cover_type', 'num_of_pages', 'rating star', 'book_length', 'book_width']]).set_index('large_cate')
        book_info = book_info.rename(columns = {'detail_cate': 'Phân loại chi tiết', 'cover_type': 'Loại bìa', 'num_of_pages': 'Số trang', 'rating star': 'Đánh giá', 'book_length': 'Độ dài', 'book_width': 'Độ rộng'})
        book_info['Loại bìa'] = book_info['Loại bìa'].apply(lambda x: 'Chưa có thông tin' if x == np.nan else x)
        book_info['Số trang'] = book_info['Số trang'].apply(convert, args = (0,))
        book_info['Đánh giá'] = book_info['Đánh giá'].apply(convert, args = (1,))
        for i in ['Độ dài', 'Độ rộng']:
            book_info[i] = book_info[i].apply(convert, args = (2,))
        st.table(book_info)

        results = choose_similar_book(selected_book)
        results = results.reset_index()
        
        st.markdown(f'## II. Gợi ý cho bạn:')
        for idx, row in results.iterrows():
            st.markdown(f'### {idx + 1} - {row.book_name}')
            price = '{:,}'.format(round(row['price']))
            st.markdown(f'Giá: {price} vnd')
            col1, col2 = st.columns(2)
            with col1:
                st.image(row['img'])
            with col2:
                detail_info = pd.DataFrame(row[['detail_cate', 'large_cate', 'cover_type', 'num_of_pages', 'rating star', 'book_length', 'book_width']]).T.set_index('large_cate')
                detail_info = detail_info.rename(columns = {'detail_cate': 'Phân loại chi tiết', 'cover_type': 'Loại bìa', 'num_of_pages': 'Số trang', 'rating star': 'Đánh giá', 'book_length': 'Độ dài', 'book_width': 'Độ rộng'})
                detail_info['Loại bìa'] = detail_info['Loại bìa'].apply(lambda x: 'Chưa có thông tin' if x == np.nan else x)
                detail_info['Số trang'] = detail_info['Số trang'].apply(convert, args = (0,))
                detail_info['Đánh giá'] = detail_info['Đánh giá'].apply(convert, args = (1,))
                for i in ['Độ dài', 'Độ rộng']:
                    detail_info[i] = detail_info[i].apply(convert, args = (2,))
                st.table(detail_info.T)
            with st.expander('Mô tả về sách'):
                st.write(row['describe'])
    for i in range(0,25):
        st.write("")


#MARKET
if options == 'BOOK MARKET':
    st.cache_data(persist = True)
    def getdata():
        books_df = pd.read_csv('data/full_info_df.csv')
        return books_df
    books_df = getdata()
    books_df = books_df.drop(['Unnamed: 0', 'wv_describe', 'clusters', 'seller', 'seller_star', 'seller_reviews_quantity', 'seller_follow', 'authentic'], axis = 1)
    books_df = books_df.rename(columns = {'img': 'Hình ảnh','detail_cate': 'Phân loại chi tiết', 'cover_type': 'Loại bìa', 'num_of_pages': 'Số trang', 'rating star': 'Đánh giá', 
                                        'book_length': 'Độ dài', 'book_width': 'Độ rộng', 'sale_quantities': 'Số lượng bán', 'discount': 'Chiết khấu', 'price': 'Giá',
                                        'describe': 'Mô tả', 'book_name': 'Tên sách', 'rating_quantity': 'Lượng đánh giá', 'publisher': 'Nhà xuất bản',
                                        'large_cate': 'Phân loại', 'publish_date': 'Ngày phát hành', 'publishing_company': 'Công ty phát hành'})
    books_df = books_df.fillna('Chưa có thông tin')

    #SIDE BAR
    st.sidebar.header('PLEASE CHOOSE YOUR INFO YOU WANT TO SEE')

    large_cate = st.sidebar.multiselect(
        'Select Large Category:',
        options = ['SELECT ALL'] + list(books_df['Phân loại'].unique()),
    )

    if 'SELECT ALL' in large_cate:
        large_cate = books_df['Phân loại'].unique()



    small_cate = st.sidebar.multiselect(
        'Select Small Category',
        options = ['SELECT ALL'] + list(books_df[books_df['Phân loại'].isin(large_cate)]['Phân loại chi tiết'].unique()),
    )

    if 'SELECT ALL' in small_cate:
        small_cate = books_df[books_df['Phân loại'].isin(large_cate)]['Phân loại chi tiết'].unique()



    cover_type = st.sidebar.multiselect(
        'Select Cover Type',
        options = ['SELECT ALL'] + list(books_df['Loại bìa'].unique()),
    )

    if 'SELECT ALL' in cover_type:
        cover_type = books_df['Loại bìa'].unique()




    #FILTER DATA
    selected_df = books_df[books_df['Phân loại'].isin(large_cate) 
                        & books_df['Phân loại chi tiết'].isin(small_cate)
                        & books_df['Loại bìa'].isin(cover_type)]

    st.title('📗 📘BOOK MARKET DESCRIPTIVE ANALYSIS 📕 📙')


    #SHOW DATA
    with st.expander("You can see or download all book data you've selected here"):
        showdata = st.multiselect('Select info you want', options = selected_df.columns, default = [])
        st.data_editor(selected_df[showdata],
                    column_config = {
                        'Hình ảnh': st.column_config.ImageColumn("Preview Image", help="Streamlit app preview screenshots")
                    })

    calculate_df = selected_df.map(lambda x: np.nan if x == 'Chưa có thông tin' else x)
    calculate_df = calculate_df.dropna()


    total_sales_quantity = calculate_df['Số lượng bán'].sum()
    total_rating_quantity = calculate_df['Lượng đánh giá'].sum()
    try:
        average_star = calculate_df['Đánh giá'].sum() / len(calculate_df['Đánh giá'])
    except:
        average_star = 0 
        average_price = 0

    info1, info2, info3 = st.columns(3, gap = 'large')

    with info1:
        st.info('Total Sales Quantity', icon = '📌')
        st.metric(label = 'Sum Sales Quantity', value = f'{total_sales_quantity:,.0f}')

    with info2:
        st.info('Total Rating Quantity', icon = '📌')
        st.metric(label = 'Sum Rating Quantity', value = f'{total_rating_quantity:,.0f}')

    with info3:
        st.info('Average Star Review', icon = '📌')
        st.metric(label = 'Average Star', value = f'{average_star:,.0f}')

    #BEST SELLER
    st.markdown('## Best Seller Book based on your filter:')
    best_seller = selected_df[['Tên sách', 'Hình ảnh', 'Số lượng bán', 'Lượng đánh giá']]
    best_seller['Số lượng bán'] = best_seller['Số lượng bán'].map(lambda x: 0 if x == 'Chưa có thông tin' else x)
    best_seller = best_seller.sort_values(by = ['Số lượng bán', 'Lượng đánh giá'], ascending = False)[:5]
    bs1, bs2, bs3, bs4, bs5 = st.columns(5)
    try:
        with bs1:
            st.image(best_seller['Hình ảnh'].iloc[0])
            st.caption(best_seller['Tên sách'].iloc[0])

        with bs2:
            st.image(best_seller['Hình ảnh'].iloc[1])
            st.caption(best_seller['Tên sách'].iloc[1])


        with bs3:
            st.image(best_seller['Hình ảnh'].iloc[2])
            st.caption(best_seller['Tên sách'].iloc[2])


        with bs4:
            st.image(best_seller['Hình ảnh'].iloc[3])
            st.caption(best_seller['Tên sách'].iloc[3])


        with bs5:
            st.image(best_seller['Hình ảnh'].iloc[4])
            st.caption(best_seller['Tên sách'].iloc[4])
    except:
        st.divider()



    #GRAPHS

    sales_by_covertype = calculate_df.groupby(by = ['Loại bìa'])['Số lượng bán'].agg('sum').sort_values()
    sales_by_cate = calculate_df.groupby(by = ['Phân loại', 'Phân loại chi tiết'])['Số lượng bán'].agg('sum').sort_values()
    avg_pages_by_large = calculate_df.groupby(by = ['Phân loại'])['Số trang'].agg('mean').sort_values()
    sales_by_company = calculate_df.groupby(by = ['Công ty phát hành'])['Số lượng bán'].agg('sum').sort_values()
    sales_by_publisher = calculate_df.groupby(by = ['Nhà xuất bản'])['Số lượng bán'].agg('sum').sort_values()
    ratequan_by_cate = calculate_df.groupby(by = ['Phân loại'])['Lượng đánh giá'].agg('sum').sort_values()



    cover_type_px = px.pie(
        sales_by_covertype,
        names = sales_by_covertype.index,
        values = 'Số lượng bán',
        title = '<b> Sales Quantity by Cover Type <b>',
        template = 'plotly_dark'
    )

    cate_px = px.treemap(
        sales_by_cate,
        path = [sales_by_cate.index.get_level_values('Phân loại'),
                sales_by_cate.index.get_level_values('Phân loại chi tiết')],
        values = 'Số lượng bán',
        title = '<b> Sales Quantity by Category <b>',
        template = 'ggplot2',
    )

    avg_pages_large = px.bar(
        avg_pages_by_large,
        x = avg_pages_by_large.index,
        y = 'Số trang',
        title = '<b> Average Num of Pages by Large Category <b>'
    )

    company_px = px.bar(
        sales_by_company,
        x = 'Số lượng bán',
        y = sales_by_company.index,
        orientation = 'h',
        title = '<b> Sales Quantity by Publishing Company <b>'
    )

    publisher_px = px.bar(
        sales_by_publisher,
        x = 'Số lượng bán',
        y = sales_by_publisher.index,
        orientation = 'h',
        title = '<b> Sales Quantity by Publisher <b>'
    )

    rate_px = px.bar(
        ratequan_by_cate,
        x = 'Lượng đánh giá',
        y = ratequan_by_cate.index,
        orientation = 'h',
        title = '<b> Rate Quantity by Large Category <b>'
    )


    cat, rate = st.columns([2, 1])
    with cat:
        st.plotly_chart(cate_px, use_container_width = True)
    with rate:
        st.plotly_chart(rate_px, use_container_width = True)


    com, pub = st.columns(2)
    with com:
        st.plotly_chart(company_px, use_container_width = True)
    with pub:
        st.plotly_chart(publisher_px, use_container_width = True)

    cov, pag = st.columns([1, 2])
    with cov:
        st.plotly_chart(cover_type_px, use_container_width = True)
    with pag:
        st.plotly_chart(avg_pages_large, use_container_width = True)

    for i in range(0, 10):
        st.markdown('')



footer = st.container()
with footer:
    e1, e2, e3 = st.columns([1,2,2])
    with e1:
        st.image('https://scontent.fsgn8-3.fna.fbcdn.net/v/t39.30808-6/300214291_923781189013440_1485100982149543062_n.jpg?_nc_cat=106&ccb=1-7&_nc_sid=5f2048&_nc_ohc=mphYeVKgA_4AX_ogF4E&_nc_ht=scontent.fsgn8-3.fna&oh=00_AfCxsJpiY_DtZPHQw5cS0CN2Yx_knRPiiQ2ikwgIl2bqBQ&oe=6562E802',
                width = 160)
    with e2:
        st.markdown('👨‍💻 Hoàng Hảo')
        st.markdown('🏢 Marketing Reseacher')
        st.markdown('🏠 Ho Chi Minh City')
        st.markdown('📞 Phone: 0866 131 594')
        st.markdown('✉️ hahoanghao1009@gmail.com')
    with e3:
        i1, i2, i3 = st.columns(3)
        with i1:
            image_url = 'https://cdn-icons-png.flaticon.com/256/174/174857.png'
            linkedin_url = "https://www.linkedin.com/in/hahoanghao1009/"

            clickable_image_html = f"""
                <a href="{linkedin_url}" target="_blank">
                    <img src="{image_url}" alt="Clickable Image" width="50">
                </a>
            """
            st.markdown(clickable_image_html, unsafe_allow_html=True)

        with i2:
            image_url = 'https://static-00.iconduck.com/assets.00/github-icon-2048x1988-jzvzcf2t.png'
            git_url = 'https://github.com/HoangHao1009'

            clickable_image_html = f"""
                <a href="{git_url}" target="_blank">
                    <img src="{image_url}" alt="Clickable Image" width="50">
                </a>
            """
            st.markdown(clickable_image_html, unsafe_allow_html=True)
        with i3:
            image_url = 'https://cdn-icons-png.flaticon.com/512/3536/3536394.png'
            fb_url = 'https://www.facebook.com/hoanghao1009/'

            clickable_image_html = f"""
                <a href="{fb_url}" target="_blank">
                    <img src="{image_url}" alt="Clickable Image" width="50">
                </a>
            """
            st.markdown(clickable_image_html, unsafe_allow_html=True)


