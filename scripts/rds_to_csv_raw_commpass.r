raw_expr_df <- readRDS("../data/CoMMpass/1stLine_Raw_ENSG.rds")

write.csv(raw_expr_df, "../data/CoMMpass/1stLine_Raw_ENSG.csv", row.names=TRUE)
