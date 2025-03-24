use yahoo_finance_api as yahoo;
use tokio;
use csv::Writer;

#[tokio::main]
async fn main() {
    let provider = yahoo::YahooConnector::new().unwrap();

    let response = provider.get_latest_quotes("NVDA", "1d")
    .await
    .expect("Failed to fetch data");

    let quotes = response.quotes().expect("failed to parse data");

    // create a csv
    let mut wtr = Writer::from_path("nvda_stock.csv")
    .expect("Failed to create CSV");

    // add header rows
    wtr.write_record(&["Date", "Open", "High", "Low", "Close", "Volume"]).expect("failed to write header");

    for quote in quotes {
        wtr.write_record(&[
            quote.timestamp.to_string(),
            quote.open.to_string(), // need to actually handle this as might not exist
            quote.high.to_string(),
            quote.low.to_string(),
            quote.close.to_string(),
            quote.volume.to_string(),
        ])
        .expect("Failed to write record");
    }

    wtr.flush().expect("failed to flush wtr");

    println!("csv saved successfully");
}