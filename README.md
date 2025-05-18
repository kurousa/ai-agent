# AI-Agent Apps

## 概要

このリポジトリは、複数のAI大規模言語モデル（LLM）を使用した各種アプリの実装を行っています。

現在利用可能なアプリは以下の通りです。

- [Chat](#chat)
- [Website Summarizer](#website-summarizer)
- [Image Recognizer](#image-recognizer)
- [Image Generator](#image-generator)

## (共通)準備

- このリポジトリは`Rye`で管理されています。未インストールの場合は先に`rye`をインストールしてください。
- `.env`ファイルを作成してください（`.env.example`からコピーすることをお勧めします）。

各サービスのAPIトークンを`.env`ファイルに設定し、プロジェクトのルートディレクトリに配置してください。環境変数は`python-dotenv`によってロードされます。

### 必要な環境変数

```shell
# OpenAI API（GPT-3.5-turbo、GPT-4o用）
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API（Claude 3.5 Haiku用）
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google API（Gemini 1.5 Flash用）
GOOGLE_API_KEY=your_google_api_key_here
```

## (共通)サポートされているモデル

現在、以下のモデルがサポートされています：

- OpenAI GPT-3.5-turbo
- OpenAI GPT-4o
- Anthropic Claude 3.5 Haiku
- Google Gemini 1.5 Flash

## Chat

チャットアプリケーションの実装です。

以下の機能を提供します：

- 複数のLLMプロバイダー（OpenAI、Anthropic、Google）のサポート
- リアルタイムのトークン使用量とコスト計算
- チャット履歴の管理
- 会話のストリーミング表示

### コスト計算

アプリケーションは、各会話でのトークン使用量を追跡し、リアルタイムでコストを計算します。

- 入力トークン（ユーザーメッセージ）と出力トークン（AI応答）別にコスト計算
- 各モデルの最新の価格情報に基づく正確なコスト計算
  - 注 Claudeについては、Claudeはトークン数を取得する方法が不明なため、1トークン=1文字として計算しているため概算値となります
- サイドバーに総コスト、入力コスト、出力コストを表示

#### トークン計算の特記事項

- OpenAIモデル：tiktokenライブラリを使用
- Geminiモデル：Google APIの組み込み関数を使用
- Claudeモデル：現在は近似値を使用（文字数ベース）

### モデル選択

サイドバーで使用するAIモデルを簡単に切り替えることができます。各モデルの特性に応じて会話を調整できます。

### チャット履歴

会話の全履歴が保存され、表示されます。「Clear Conversation」ボタンでいつでも履歴をクリアできます。

### 使用方法

- チャットアプリを実行：

  ```shell
  rye run chat
  ```

- アプリケーション起動後：
  1. サイドバーからモデルを選択
  2. 必要に応じてtemperature値を調整（0.0-1.0）
  3. チャットボックスにメッセージを入力して会話を開始
  4. サイドバーでコスト情報を確認

## Website Summarizer

ウェブサイト要約アプリケーションの実装です。

指定されたURLのウェブサイトの内容を要約します。

### 使用方法

- ウェブサイト要約アプリを実行：

  ```shell
  rye run website_summarizer
  ```

- アプリケーション起動後：
  1. URL入力ボックスに要約したいウェブサイトのURLを入力し、Enterキーを押下
  2. 要約結果が表示されます

### 仕組み

1.  URLからウェブサイトのコンテンツをスクレイピング
2.  スクレイピングしたコンテンツをLLMに送信
3.  LLMがコンテンツを要約
4.  要約結果を画面に表示
    - 要約にはOpenAIのGPT-3.5-turboが使用されます。

### 必要なライブラリ

- `requests`: ウェブサイトのコンテンツを取得するために使用
- `beautifulsoup4`: HTMLコンテンツの解析に使用
- `streamlit`: UIの構築に使用
- `openai`: OpenAI APIとの連携に使用
- `tiktoken`: OpenAI APIで利用するトークン数の計算に使用

これらのライブラリは、`rye`によって管理されています。

## Image Recognizer

アップロードされた画像ファイルについての説明を行うアプリケーション。

### 使用方法

```shell
rye run image_recognizer
```

### 仕組み

1. 画像アップロードしてもらう
2. ユーザーに聞きたいことを入力
3. 画像とユーザーからのインプットを基にプロンプトを組み上げて、LLMへ問い合わせ
   - OpenAI GPT-4oで処理
4. 処理結果を画面に出力

## Image Generator

アップロードされた画像ファイルを基に、ユーザーにリクエストに応じた画像を、
DALL-E 3で生成する

### 使用方法

```shell
rye run image_generator
```

### 仕組み

1. 画像アップロードしてもらう
2. ユーザーにリクエストを入力してもらう
3. 画像とユーザーからのインプットを基に、GPT-4VにDALL-E 3プロンプトを組み上げる
4. DALL-E 3へ生成したプロンプトを投げる
5. 処理結果を画面に出力

## 開発者向け情報

### (ChatApp)モデルの追加方法

新しいモデルを追加するには：

1. `MODEL_PRICE`オブジェクトに価格情報を追加
2. `select_model`関数に新モデルの設定を追加
3. 必要に応じて`get_message_counts`関数でトークン計算方法を調整
