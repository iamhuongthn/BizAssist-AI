"""40 distinct project-authored POS knowledge entries; not external citations."""
import pandas as pd
TOPICS = [
('Gross sales','Gross sales sum quantity multiplied by unit price for completed order items before discounts or refunds.'),
('Net sales','Net sales are gross sales minus order-level discounts and refunds, each deducted once.'),
('Average order value','AOV is net sales divided by the count of completed orders. Return N/A if there are no completed orders.'),
('Eligible orders','Completed orders are eligible, even when partially or fully refunded; cancelled and failed orders are excluded.'),
('Cancelled orders','Cancelled orders are not included in gross sales, net sales, order counts or product rankings.'),
('Failed orders','Failed orders are excluded from sales calculations and eligible order counts.'),
('Refund treatment','Refunds reduce net sales but do not remove a completed order from eligible order counts.'),
('Partial refund','A partial refund subtracts only its refunded amount from net sales.'),
('Full refund','A full refund can reduce an order net amount to zero without changing completed-order eligibility.'),
('Discount treatment','Order-level discounts reduce net sales once per order, not once per item.'),
('Gross item revenue','Product gross item revenue sums quantity times unit price across completed order items.'),
('Units sold','Units sold sum item quantities from completed orders for each product.'),
('Product rankings by units','Rank products by summed units sold and explicitly state the metric and date range.'),
('Product rankings by revenue','Rank products by gross item revenue, before order-level discounts and refunds.'),
('Refund allocation','This prototype does not allocate order-level refunds to individual products.'),
('Daily reporting','Daily reports group completed orders by the order timestamp calendar date.'),
('Reporting period','Specify start and end dates when requesting product rankings over a date range.'),
('Zero order days','A day without completed orders has zero gross and net sales and an unavailable AOV.'),
('Order identifier','order_id uniquely identifies an order and links its line items.'),
('Item identifier','item_id uniquely identifies a POS line item.'),
('Product identifier','product_id identifies a catalog product referenced by order items.'),
('Foreign keys','Each order item must reference an existing order and product.'),
('Duplicate identifiers','Repeated primary keys are invalid and must be flagged rather than silently aggregated.'),
('Missing timestamps','An order with a missing or invalid timestamp cannot be assigned to a reporting date.'),
('Invalid quantities','Completed-sale item quantities must be positive to support valid unit and revenue totals.'),
('Invalid prices','Unit prices cannot be negative; invalid rows should be quarantined.'),
('Financial reconciliation','Discounts and refunds cannot jointly exceed the completed order gross amount.'),
('Order status','Supported order statuses are completed, cancelled and failed.'),
('Data provenance','BizAssist uses generated synthetic POS records, not live store transactions.'),
('Privacy','Synthetic records do not contain real customer identities or payment card data.'),
('Data validation','Validate required columns, types, keys, values and financial totals before analytics.'),
('Rejected records','Quarantined records retain a reason for rejection and are excluded from calculations.'),
('Categorical encoding','One-hot encoding converts order status and product category into numeric indicator columns.'),
('Numerical scaling','Standardization centers numerical analysis features and scales by training-set standard deviation.'),
('Day of week','day_of_week derives from an order timestamp and can reveal weekly demand patterns.'),
('Hour of day','hour_of_day derives from an order timestamp and can reveal time-of-day demand patterns.'),
('Knowledge retrieval','FAQ documents are retrieved using text similarity and reported with source IDs.'),
('Retrieval uncertainty','Weakly matched knowledge questions should receive a limitation instead of a fabricated answer.'),
('No forecasting','The prototype reports historical synthetic sales and does not predict future profits.'),
('No payment processing','The prototype cannot process card payments, payroll or tax filings.')]
def build_knowledge(path):
    assert len(TOPICS)==40
    df=pd.DataFrame([{'doc_id':f'KB{i:03d}','title':t,'content':c,'source':'Project-authored POS metric dictionary (synthetic demonstration)','updated_at':'2026-11-09'} for i,(t,c) in enumerate(TOPICS,1)])
    df.to_csv(path,index=False)
    return df
