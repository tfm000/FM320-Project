%%  FM320 Risk Management Course Project - Candidate Number: 40946 %%

%% Clearing All Prior MATLAB Data %% 

clc;                    % Clears command window
clear all;              % Removes all variables from workspace
clf;                    % Clears current figure window
close all;              % Close all windows

%% Importing the Data %%

Raw_Data                =   xlsread('Project Data.xlsx','FM320-FM321 - Project Data','E9:Z7546'); %Includes Dates
NDates                  =   size(Raw_Data,1);
NStocks                 =   size(Raw_Data,2)-2 ;
Tickers                 =   {'MSFT ','XOM ','FB ','CVX ','GOOG ','APPL ','PFE ','JNJ ','WFC ','JPM ','WMT ','BAC ','VZ ','T ','HD ','AMZN ','GOOGL ','MA ','UNH ','V ','SPX '}; 
%% Calculating Log Returns and Statistics %%

Log_Stock_Returns       =   Raw_Data(:,1:(end-1));
Dates_for_log_returns   =   Raw_Data(:,22);
Days_In_Year            =   252;
Mean_Log_Returns        =   mean(Log_Stock_Returns,'omitnan');
Std_Log_Returns         =   std(Log_Stock_Returns,'omitnan');
Annualised_Mean_Log_Ret =   Mean_Log_Returns*Days_In_Year;
Annualised_Std_Log_Ret  =   Std_Log_Returns*sqrt(Days_In_Year);

% Producing a Display For Our Stock Data %
StockData              =   [Mean_Log_Returns;Annualised_Mean_Log_Ret;Std_Log_Returns;Annualised_Std_Log_Ret];
RowHeaders             =   {'Mean Log Returns', 'Mean Log Returns (Ann.)', 'Std Log Returns', 'Std Log Returns (Ann.)'};
ColHeaders             =   [{' '} Tickers];
StockData2             =   [RowHeaders' num2cell(StockData)];
StockDisplay           =   [ColHeaders; StockData2];

%% Creating a Universe of Our 2 Securities %%

universe                      =  [1 10];
universe_Log_Returns          =  Log_Stock_Returns(:,universe);
Squared_Log_Returns           =  Log_Stock_Returns .* Log_Stock_Returns;
universe_Squared_Log_Returns  =  Squared_Log_Returns(:,universe);
UniverseTickers               =  Tickers(universe);

% Other Global Data %
StartDate     =  2000;
EndDate       =  2020;
xaxis         =  linspace(StartDate,EndDate,NDates);


%% Calculating Optimal Portfolio Weights for Our Securities %%

% Quadratic Optimisation Using Unconditional Sigma % 
Sigma                     =  cov(universe_Log_Returns);
Sigma_Annualised          =  Sigma * Days_In_Year;
Av_Log_Returns            =  mean(universe_Log_Returns);
Av_Annualised_Log_Returns =  Av_Log_Returns*Days_In_Year;
Denominator               =  sum(Sigma_Annualised * Av_Annualised_Log_Returns');
Numerator                 =  inv(Sigma_Annualised) * Av_Annualised_Log_Returns';
Quadratic_Optimal_Weights =  Numerator / Denominator;

% We See However, This Yields an Extreme Optimal Weight For MSFT, Thus
% Another Method Should be Used To Assign Optimal Weights. 

% As the number of assets is small (2), it is feasible to use monte-carlo
% simulation to randomly assign asset's portfolio weights, and then select
% the one with the greatest sharpe ratio. Note: For a greater number of
% assets, such a method is not feasible, and Mathematical Optimisation must
% be performed.

% Sharpe Ratio Portfolio Optimisation %
n           =  50000; % Number of Simulations
all_weights =  NaN(n,2);
ERs         =  NaN(n,1);
vols        =  NaN(n,1);
SRs         =  NaN(n,1);

for i = 1:n
    weights          =  rand(2,1);
    weights          =  weights/sum(weights); % So Portfolio Weights Sum to 1 (no risk free asset)
    all_weights(i,:) =  weights;
    ERs(i)           =  Av_Annualised_Log_Returns * weights;
    vols(i)          =  sqrt( weights' * Sigma_Annualised * weights );
    SRs(i)           =  ERs(i)/vols(i); % Assuming a risk free rate of 0 
end
[max_SharpeRatio, argmax_SharpeRatio] =  max(SRs); % Finding the Weights Corresponding to the greatest sharpe ratio
Optimal_Weights                       =  all_weights(argmax_SharpeRatio,:);
Optimal_Weights                       =  Optimal_Weights';

% Visualising The MC Simulated Weights %
figure
grid on;
hold on;
scatter(vols,ERs,[],SRs,'filled');
scatter(vols(argmax_SharpeRatio),ERs(argmax_SharpeRatio),'MarkerEdgeColor','black','MarkerFaceColor','r','LineWidth',2);
xlabel('Portfolio Volatility');
ylabel('Portfolio Expected Return');
title('Efficency Frontier of MSFT + JPM Portfolios');

% We see these optimal weights are much more reasonable, and thus we shall
% use them in our analysis.

% Calculating Portfolio Returns Using Optimal Weights %
Log_Portfolio_Returns = Optimal_Weights(1)*universe_Log_Returns(:,1) + Optimal_Weights(2)*universe_Log_Returns(:,2); 
Squared_Log_Portfolio_Returns = Log_Portfolio_Returns .* Log_Portfolio_Returns;

%% Multivariate Models %%
%% Producing a DCC Model, Using GARCH, for Our Chosen Securities: MSFT & JPM %%

% Estimating DCC Model (takes a while) %
[DCC_Parameters]               =  dcc(universe_Log_Returns,[],1,0,1);

% Constructing DDC Model Matrices %
Omega                          =  [DCC_Parameters(1) DCC_Parameters(4)]; %GARCH(1,1) Omegas for each security
Alpha                          =  [DCC_Parameters(2) DCC_Parameters(5)]; %GARCH(1,1) ARCH params for each security
Beta                           =  [DCC_Parameters(3) DCC_Parameters(6)]; %GARCH(1,1) GARCH params for each security
R_Bar                          =  [1 DCC_Parameters(7); DCC_Parameters(7) 1];
a                              =  DCC_Parameters(8); % DCC Alpha
b                              =  DCC_Parameters(9); % DCC Beta
% Note: R_Bar is the unconditional correlation matrix estimated by the DCC model

% Computing GARCH Volatilities and Standard Residuals For Each Asset in Our Universe %
GARCH_Variance                 =  NaN(NDates,2);
GARCH_Variance(1,:)            =  Omega ./ (1 - Alpha -Beta); % Initialising the GARCH volatilities with the unconditional variance

for i=2:NDates
    GARCH_Variance(i,:)        =  Omega + Alpha .* universe_Squared_Log_Returns(i-1,:) + ...
                                  Beta .* GARCH_Variance(i-1,:);
end

GARCH_Volatility               =  sqrt(GARCH_Variance);
GARCH_Standardised_Residuals   =  universe_Log_Returns ./ GARCH_Volatility;

% Computining Values For Q_t and R_t %
DCC_GARCH_Qt                   =  NaN(2,2,NDates); % 3D matrix, the first two dimensions will be filled
DCC_GARCH_Rt                   =  NaN(2,2,NDates); % with correlation matrices, and the third being time.
DCC_GARCH_Qt(:,:,1)            =  R_Bar; %Initialising using R_Bar
DCC_GARCH_Rt(:,:,1)            =  R_Bar; 

for i=2:NDates            %Iterating over time, using recursive formula to produce correlation matrices
    DCC_GARCH_Qt(:,:,i)        =  (1 - a - b) * R_Bar + ... 
                                  ( a * GARCH_Standardised_Residuals(i-1,:)' * GARCH_Standardised_Residuals(i-1,:) ) + ... 
                                  ( b * DCC_GARCH_Qt(:,:,i-1) );
    Aux                        =  [sqrt(DCC_GARCH_Qt(1,1,i)) sqrt(DCC_GARCH_Qt(2,2,i))];
    DCC_GARCH_Rt(:,:,i)        =  DCC_GARCH_Qt(:,:,i) ./ (Aux' * Aux);
end

% Extracting DCC Volatility and Correlation Estimates For MSFT & JPM %
DCC_GARCH_Vol_MSFT             =  GARCH_Volatility(:,1);
DCC_GARCH_Vol_JPM              =  GARCH_Volatility(:,2);
DCC_GARCH_Corr                 =  DCC_GARCH_Rt(1,2,:);
DCC_GARCH_Corr                 =  reshape(DCC_GARCH_Corr,NDates,1);

% Producing Sigma_t and GARCH-DCC Portfolio Volatility Estimates %
D_t_GARCH                      =  NaN(2,2,NDates);
Sigma_t_GARCH                  =  NaN(2,2,NDates);
GARCH_DCC_PortfolioVariance_t  =  NaN(NDates,1);
for j = 1:NDates
    % Calculating D_t %
    D_t_GARCH(:,:,j)           =  [DCC_GARCH_Vol_MSFT(j) 0; 0 DCC_GARCH_Vol_JPM(j)];  
    
    % Estimating Sigma_t %
    Sigma_t_GARCH(:,:,j)       =  D_t_GARCH(:,:,j) * DCC_GARCH_Rt(:,:,j) * D_t_GARCH(:,:,j);
    
    % Estimating Conditional Portfolio Variance %
    GARCH_DCC_PortfolioVariance_t(j) =  Optimal_Weights' * Sigma_t_GARCH(:,:,j) * Optimal_Weights;
end 
GARCH_DCC_PortfolioVolatility_t =  sqrt(GARCH_DCC_PortfolioVariance_t);

%% Producing a DCC Model, Using GJR-GARCH, For Our Chosen Securities: MSFT & JPM %%

% Computing GJR-GARCH Volatilities and Standard Residuals For Each Asset in Our Universe %
GJR_Parameters =  NaN(6,2);
GJR_Var        =  NaN(NDates,2);
for i = 1:2
    % Estimating a GJR-GARCH Model For Each Security % 
    Security = Log_Stock_Returns(:,universe(i));
    [Parameters, LogL, ~, VCov] = tarch(Security, 1,1,1);
    GJR_Parameters(1,i)         = Parameters(1,1); % Omega
    GJR_Parameters(2,i)         = Parameters(2,1); % ARCH Coefficient
    GJR_Parameters(3,i)         = Parameters(4,1); % GARCH Coefficient
    GJR_Parameters(4,i)         = Parameters(3,1); % Leverage Coefficient
    
    % Conducting Parameter Tests on the Leverage Coefficient %
    Standard_Error          = sqrt(diag(VCov));
    Leverage_Standard_Error = Standard_Error(3); 
    Leverage_Test_Statistic = GJR_Parameters(4,i)/Leverage_Standard_Error;
    Leverage_P_Value        = 2*(1-normcdf(Leverage_Test_Statistic));
    GJR_Parameters(5,i)     = Leverage_Test_Statistic;
    GJR_Parameters(6,i)     = Leverage_P_Value;
    
    % Producing GJR-GARCH Volatility Esimates %
    GJR_Var(1,i) = var(Security); % Initialising Using Variance of Log Returns
    for j = 2:NDates
        if Security(j-1) > 0
            GJR_Var(j,i) = GJR_Parameters(1,i) + GJR_Parameters(2,i)*universe_Squared_Log_Returns(j-1,i) +... 
                           GJR_Parameters(3,i)*GJR_Var(j-1,i);
        else
            GJR_Var(j,i) = GJR_Parameters(1,i) + GJR_Parameters(3,i)*GJR_Var(j-1,i) + ...
                          ( GJR_Parameters(2,i) + GJR_Parameters(4,i) )*universe_Squared_Log_Returns(j-1,i);
        end
    end
end
GJR_Vol = sqrt(GJR_Var);
GJR_Standardised_Residuals =  universe_Log_Returns ./ GJR_Vol;

% Creating a Display For Our Leverage Test Statistics %
CellData            = GJR_Parameters(5:6,:);
RowHeaders          = {'Test Statistic','P-Value'};
ColHeaders          = {'Leverage GJR-GARCH Data For Security: ','MSFT','JPM'};
DisplayData         = [RowHeaders' num2cell(CellData)];
Leverage_Display    = [ColHeaders; DisplayData]; 

% Computining Values For Q_t and R_t %
DCC_GJR_Qt            =  NaN(2,2,NDates); % 3D matrix, the first two dimensions will be filled
DCC_GJR_Rt            =  NaN(2,2,NDates); % with correlation matrices, and the third being time.
DCC_GJR_Qt(:,:,1)     =  R_Bar; %Initialising using R_Bar
DCC_GJR_Rt(:,:,1)     =  R_Bar; % Conditional Correlation Matrices, using GJR-GARCH Standardised Residuals
for i=2:NDates            %Iterating over time, using recursive formula to produce correlation matrices
    DCC_GJR_Qt(:,:,i) =  (1 - a - b) * R_Bar + ... 
                              ( a * GJR_Standardised_Residuals(i-1,:)' * GJR_Standardised_Residuals(i-1,:) ) + ... 
                              ( b * DCC_GJR_Qt(:,:,i-1) );
    Aux               =  [sqrt(DCC_GJR_Qt(1,1,i)) sqrt(DCC_GJR_Qt(2,2,i))];
    DCC_GJR_Rt(:,:,i) =  DCC_GJR_Qt(:,:,i) ./ (Aux' * Aux);
end

% Extracting DCC Volatility and Correlation Estimates For MSFT & JPM %
DCC_GJR_Vol_MSFT =  GJR_Vol(:,1);
DCC_GJR_Vol_JPM  =  GJR_Vol(:,2);
DCC_GJR_Corr     =  DCC_GJR_Rt(1,2,:);
DCC_GJR_Corr     =  reshape(DCC_GJR_Corr,NDates,1);

% Producing Sigma_t and GJR-DCC Portfolio Volatility Estimates %
D_t_GJR                     =  NaN(2,2,NDates);
Sigma_t_GJR                 =  NaN(2,2,NDates);
GJR_DCC_PortfolioVariance_t = NaN(NDates,1);
for j = 1:NDates
    % Calculating D_t %
    D_t_GJR(:,:,j)          = [DCC_GJR_Vol_MSFT(j) 0; 0 DCC_GJR_Vol_JPM(j)];  
    
    % Estimating Sigma_t %
    Sigma_t_GJR(:,:,j)      = D_t_GJR(:,:,j) * DCC_GJR_Rt(:,:,j) * D_t_GJR(:,:,j);
    
    % Estimating Conditional Portfolio Variance %
    GJR_DCC_PortfolioVariance_t(j) = Optimal_Weights' * Sigma_t_GJR(:,:,j) * Optimal_Weights;
end 
GJR_DCC_PortfolioVolatility_t      = sqrt(GJR_DCC_PortfolioVariance_t);

%% Producing an Orthogonal GARCH Model For Securities: MSFT & JPM %%

% Performing PCA Calculations %
[PCA_Portfolio_Weights, ~,Factor_Variance] =   pca(universe_Log_Returns); 
Total_Variance                             =   sum(Factor_Variance); %Calculating the total variance of the securities 
Explained_Variance_Fraction                =   Factor_Variance / Total_Variance; %Calculating the fraction of the total variance explained by each factor portfolio

% Producing PCA Display %
SecurityHeader                             =   [{' '} UniverseTickers];
RowHeaders                                 =   [SecurityHeader(2:end) {'Variance','Pct. Total Var'}];
ColHeaders                                 =   {'Principle Component','1','2'};
PCA_Data                                   =   [PCA_Portfolio_Weights'; Factor_Variance' ; Explained_Variance_Fraction'];
DisplayData                                =   [RowHeaders' num2cell(PCA_Data)];
PCA_Display_for_OGarch                     =   [ColHeaders; DisplayData]; %A display housing our PCA data for 
                                               %the 2 securities we will be performing O-GARCH estimation on
                                          
% The above PCA analysis, seems to indicate a single common factor: 
% The Market, to which both stocks are exposed. The second portfolio is
% represents the idiosyncratic risks associated with the JPM & MSFT stocks.

% Producing a Multivariate O-GARCH(1,1) Model %
NPCA_Factors               = 1;
[~, Sigma_t_OGARCH, W, PC] = o_mvgarch(universe_Log_Returns, NPCA_Factors,1,0,1);
GarchParameterValues       = NaN(5,2);
GarchVolatility            = NaN(NDates,2);
O_GARCH_Volatilities       = NaN(NDates,2);

% Producing O-GARCH(1,1) Conditional Portfolio Variance Estimates %
OGARCH_PortfolioVariance_t = NaN(NDates,1);
for j = 1:NDates
    OGARCH_PortfolioVariance_t(j) = Optimal_Weights' * Sigma_t_OGARCH(:,:,j) * Optimal_Weights;
end
OGARCH_PortfolioVolatility_t      = sqrt(OGARCH_PortfolioVariance_t); 

% Producing O-GARCH(1,1) Correlation Estimates For Analysis %
for i = 1:2
    % Extracting O-GARCH Volatilities %
    O_Garch_Vol               = sqrt( squeeze(Sigma_t_OGARCH(i,i,:)) );
    O_GARCH_Volatilities(:,i) = O_Garch_Vol;
end    
MSFT_var            =  squeeze(Sigma_t_OGARCH(1,1,:));
JPM_var             =  squeeze(Sigma_t_OGARCH(2,2,:));
MSFT_JPM_covariance = squeeze(Sigma_t_OGARCH(1,2,:));
O_GARCH_Correlation = MSFT_JPM_covariance ./ sqrt(MSFT_var .* JPM_var);

%% Univariate Models %%
%% Producing a GARCH(1,1) Model For Portfolio Log Returns %%

[Parameters,PortfolioGarch_LogL,~,VCV] = tarch(Log_Portfolio_Returns,1,0,1);
GarchValues                            = NaN(5,1);
for i = 1:3
    GarchValues(i,1) = Parameters(i,1); % Storing Omega, ARCH and GARCH Coeffiecients respectfully
end
GarchValues(4) = sum(GarchValues(2:3)); % Alpha + Beta
GarchValues(5) = GarchValues(1)/(1-GarchValues(4)); % Unconditional Model Variance
ParameterStandardErrors = sqrt(diag(VCV)); 

% Testing Parameter Significance %
TestStats = NaN(5,1);
P_Values  = NaN(5,1);
for i = 1:3
    Parameter              = GarchValues(i);
    ParameterStandardError = ParameterStandardErrors(i);
    TestStats(i)           = Parameter./ParameterStandardError;
    P_Values(i)            = 2*(1-normcdf(TestStats(i)));
end

% Producing a Display For Our GARCH Data %
CellData     = [GarchValues TestStats P_Values];
RowHeaders   = {'Omega','ARCH(1)','GARCH(1)','Sum of Coeffs.','Unconditonal Variance'};
ColHeaders   = {' ','Portfolio Log Returns','Parameter Test Statistic','Paramater P-Value'};
DisplayData  = [RowHeaders' num2cell(CellData)];
GARCHDisplay = [ColHeaders; DisplayData];

% Computing Model Estimates For Portfolio Log Return Volatility %
GARCH_PortfolioVariance_t    = NaN(NDates,1);
GARCH_PortfolioVariance_t(1) = GarchValues(5); % Initialising Using Unconditional Model Variance
for j = 2:NDates
    GARCH_PortfolioVariance_t(j) = GarchValues(1) + GarchValues(2) .* Squared_Log_Portfolio_Returns(j-1) + ...
                                   GarchValues(3) .* GARCH_PortfolioVariance_t(j-1);
end
GARCH_PortfolioVolatility_t = sqrt(GARCH_PortfolioVariance_t);

%% Producing a GJR-GARCH(1,1,1) Model For Portfolio Log Returns %%

% Estimating a GJR-GARCH(1,1,1) Model For Portfolio Log Returns %
[Parameters,PortfolioGJRGARCH_LogL,~,VCov] = tarch(Log_Portfolio_Returns,1,1,1);
GJRGARCHValues                             = NaN(4,1);
GJRGARCHValues(1)                          = Parameters(1,1); % Omega
GJRGARCHValues(2)                          = Parameters(2,1); % ARCH(1) Coefficient
GJRGARCHValues(3)                          = Parameters(4,1); % GARCH(1) Coefficient
GJRGARCHValues(4)                          = Parameters(3,1); % Leverage Coeffieicent

% Conducting a Log Likelihood Statistical Test For Leverage %
LogLikelihood_Test_Stat = -2*(PortfolioGarch_LogL - PortfolioGJRGARCH_LogL);
LogLikelihood_P_Value   = (1 - chi2cdf(LogLikelihood_Test_Stat,1));

% Conducting Parameter Tests For GJR-GARCH(1,1,1) %
GJRParameterStandardErrors = sqrt(diag(VCov));
TestStats                  = NaN(4,1);
P_Values                   = NaN(4,1);
for i = 1:4
    Parameter              = GJRGARCHValues(i);
    ParameterStandardError = GJRParameterStandardErrors(i);
    TestStats(i)           = Parameter ./ ParameterStandardError;
    P_Values(i)            = 2*(1-normcdf(TestStats(i)));
end

% Producing a Display For Our GJR-GARCH(1,1,1) Data %
LogL_Cell_Data      = NaN(4,2);
LogL_Cell_Data(4,:) = [LogLikelihood_Test_Stat LogLikelihood_P_Value];
CellData            = [GJRGARCHValues TestStats P_Values LogL_Cell_Data];
RowHeaders          = {'Omega','ARCH(1)','GARCH(!)','Leverage'};
ColHeaders          = {'GJR-GARCH Parameters','Values','Parameter Test Statistic','Parameter P-Value','LogL Test Statistic','LogL P-Value'};
DisplayData         = [RowHeaders' num2cell(CellData)];
GJR_GARCH_Display   = [ColHeaders; DisplayData]; 

% Producing GJR-GARCH(1,1,1) Volatility Estimates %
GJR_PortfolioVariance_t    = NaN(NDates,1);
GJR_PortfolioVariance_t(1) = var(Log_Portfolio_Returns); % Initialising Using Unconditional Sample Variance
for j = 2:NDates
    if Log_Portfolio_Returns(j-1) > 0
        GJR_PortfolioVariance_t(j) = GJRGARCHValues(1) + GJRGARCHValues(2) .* Squared_Log_Portfolio_Returns(j-1) +...
                                     GJRGARCHValues(3) .* GJR_PortfolioVariance_t(j-1);
    else
        GJR_PortfolioVariance_t(j) = GJRGARCHValues(1) + GJRGARCHValues(3) .* GJR_PortfolioVariance_t(j-1) +...
                                     ( GJRGARCHValues(2) + GJRGARCHValues(4) ) .* Squared_Log_Portfolio_Returns(j-1);
    end
end
GJR_PortfolioVolatility_t = sqrt(GJR_PortfolioVariance_t);

%% Comparing Models %%
%% Creating Estimates For Pairwise Correlations, Using a Rolling Windows of 25 and 100 Days %

% 25 Day RW %
window_length                                    =  25;
Rolling_Window_25_Correlation_Estimates          =  NaN(NDates,1);

for i=window_length:NDates
    lower_range = i+1-window_length;
    upper_range = i;
    rolling_window_returns                       = universe_Log_Returns(lower_range:upper_range,:);
    rolling_window_corr_matrix                   = corr(rolling_window_returns);
    Rolling_Window_25_Correlation_Estimates(i,1) = rolling_window_corr_matrix(1,2);
end

% 100 Day RW %
window_length                                    =  100;
Rolling_Window_100_Correlation_Estimates         =  NaN(NDates,1);

for i=window_length:NDates
    lower_range = i+1-window_length;
    upper_range = i;
    rolling_window_returns                        = universe_Log_Returns(lower_range:upper_range,:);
    rolling_window_corr_matrix                    = corr(rolling_window_returns);
    Rolling_Window_100_Correlation_Estimates(i,1) = rolling_window_corr_matrix(1,2);
end

%% Comparing Multivariate Models %%

% Comparing Multivariate Model Correlation Estimates %
Rolling_Windows = [Rolling_Window_25_Correlation_Estimates Rolling_Window_100_Correlation_Estimates];
MV_Correlations = [DCC_GARCH_Corr DCC_GJR_Corr O_GARCH_Correlation];
ModelNames      = {'DCC-GARCH','DCC-GJR-GARCH','O-GARCH'};
Colours         = {'red','blue'};
figure
for i = 1:3
    subplot(3,1,i) 
    hold on;
    grid on;
    for j = 1:2
        plot(xaxis,Rolling_Windows(:,j),'LineStyle','-','LineWidth',1,'Color',char(Colours(j)));
    end
    plot(xaxis,MV_Correlations(:,i),'LineStyle','-','LineWidth',1,'Color','black');
    Title = [ModelNames(i) 'Correlation Estimates'];
    title(Title);
    xlabel('Dates');
    ylabel('Correlation');
    legend({'25 Day RW','100 Day RW',char(ModelNames(i))},'Location','best');
    ylim([-0.5,1]);
end

% Using the 100 Day Correlation Rolling Window as a Baseline, we see the
% DCC models provide much better conditional correlation estimates than the
% O-GARCH model, which persistently overestimates correlation relative to
% the baseline, only producing accurate estimates durring and briefly after
% crisis periods.

% Comparing Multivariate Model Stock Volatility Estimates %
MV_Stock_Volatility_Estimates = [GARCH_Volatility GJR_Vol O_GARCH_Volatilities];
Colours2                      = {'red','red','blue','blue','yellow','yellow'};
for i = 1:2
    % Plotting All Estimates on a Single Graph %
    figure
    hold on;
    grid on;
    plot(xaxis,universe_Log_Returns(:,i),'LineStyle','-','LineWidth',1,'Color','black');
    if i == 1
        for j = 1:2:6
            % MSFT %
            plot(xaxis,2*MV_Stock_Volatility_Estimates(:,j),'LineStyle','-','LineWidth',1,'Color',char(Colours2(j)));
        end 
        for j = 1:2:6
            plot(xaxis,-2*MV_Stock_Volatility_Estimates(:,j),'LineStyle','-','LineWidth',1,'Color',char(Colours2(j)));
        end
    else
        for j = 2:2:6
            % JPM %
            plot(xaxis,2*MV_Stock_Volatility_Estimates(:,j),'LineStyle','-','LineWidth',1,'Color',char(Colours2(j)));
        end
        for j = 2:2:6
             plot(xaxis,-2*MV_Stock_Volatility_Estimates(:,j),'LineStyle','-','LineWidth',1,'Color',char(Colours2(j)));
        end
    end
    Title =  ['Multivariate +/-2 * Volatility Estimates of ' char(UniverseTickers(i)) 'Log Returns'];
    title(Title);
    xlabel('Dates');
    ylabel('Volatility');
    legend({'Log Returns','DCC-GARCH','DCC-GJR-GARCH','O-GARCH'},'Location','best');
end
ModelNames2 = {'DCC-GARCH','DCC-GARCH','DCC-GJR-GARCH','DCC-GJR-GARCH','O-GARCH','O-GARCH'};
figure
for i = 1:6
    subplot(3,2,i)
    hold on;
    grid on;
    if mod(i,2) == 0
        % JPM %
        Title =  ['+/-2 *' char(ModelNames2(i)) ' Volatility Estimates of ' char(UniverseTickers(2)) 'Log Returns'];
        plot(xaxis,universe_Log_Returns(:,2),'LineStyle','-','LineWidth',1,'Color','black');
        ylim([min(universe_Log_Returns(:,2))-0.01,max(universe_Log_Returns(:,2))+0.01])
    else
        % MSFT %
        Title =  ['+/-2 *' char(ModelNames2(i)) ' Volatility Estimates of ' char(UniverseTickers(1)) 'Log Returns'];
        plot(xaxis,universe_Log_Returns(:,1),'LineStyle','-','LineWidth',1,'Color','black');
    end
    plot(xaxis,2*MV_Stock_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color','red');
    plot(xaxis,-2*MV_Stock_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color','red');
    title(Title);
    xlabel('Dates');
    ylabel('Volatility');
    legend({'Log Returns' char(ModelNames2(i))},'Location','best');
end

% Comparing Multivariate Model Conditional Portfolio Volatility Estimates %
MV_Portfolio_Volatility_Estimates = [GARCH_DCC_PortfolioVolatility_t GJR_DCC_PortfolioVolatility_t OGARCH_PortfolioVolatility_t];
figure
for i = 1:3
    subplot(3,1,i)
    grid on;
    hold on;
    plot(xaxis,Log_Portfolio_Returns,'LineStyle','-','LineWidth',1,'Color','black');
    plot(xaxis,2*MV_Portfolio_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color','red');
    plot(xaxis,-2*MV_Portfolio_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color','red');
    Title = ['+/-2*' char(ModelNames(i)) ' Volatility Estimates for Log Portfolio Returns'];
    title(Title);
    xlabel('Dates');
    ylabel('Portfolio Volatility');
    legend({'Log Portfolio Returns' char(ModelNames(i))},'Location','best');
    ylim([min(Log_Portfolio_Returns)-0.01,max(Log_Portfolio_Returns)+0.01]);
end

% We see that the O-GARCH volatility estimates for MSFT are incredibly
% poor, massively overstimating vol durring calm markets with channel like
% predictions, whilst the DCC provide very good estimates. For this reason as well as the correlation comments made
% above it is reasonable to conclude that the DCC type models are superior.
% Whilst the distinction between the estimates of DCC-GARCH and
% DCC-GJR-GARCH are not immediately apparent, DCC-GJR-GARCH does seem to
% produce volatility estimates which fit the observed results more
% accurately, and overestimates the overall volatilitly less that
% DCC-GARCH, primarily due to having a smaller omega parameter in both
% models, and thus a lower volatility lower bound. For this reason, as well
% as its theoretical properties (i.e. being able to encorporate the
% leverage effect), it is reasonable to conclude that the DCC-GJR-GARCH
% model is the superior multivariate model.

%% Comparing Univariate Models %%

UV_Model_Volatility_Estimates = [GJR_PortfolioVolatility_t GARCH_PortfolioVolatility_t ];
figure
hold on;
grid on;
plot(xaxis,Log_Portfolio_Returns,'LineStyle','-','LineWidth',1,'Color','black');
for i = 1:2
    plot(xaxis,2*UV_Model_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color',char(Colours(i)));
    plot(xaxis,-2*UV_Model_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color',char(Colours(i)));
end
legend({'Portfolio Log Returns','2*GJR-GARCH(1,1,1)','-2*GJR-GARCH(1,1,1)','2*GARCH(1,1)','-2*GARCH(1,1)'},'Location','best');
ylim([min(Log_Portfolio_Returns)-0.01,max(Log_Portfolio_Returns)+0.01]);
title('+/-2 * Univariate Conditional Portfolio Volatilility Estimates');
xlabel('Dates');
ylabel('Portfolio Volatility');

% We Note that whilst the estimates from the GARCH and GJR-GARCH models are
% extremely similar, GJR-GARCH(1,1,1) produces greater estimates
% for shocks during stress events when returns are negative, and thus would
% serve as a better model for hedging positions during crises than
% GARCH(1,1). Thus it is reasonable to conclude that GJR-GARCH is the
% better univariate model for this portfolio.

%% Comparing Univariate GJR-GARCH(1,1,1) & Multivariate DCC-GJR-GARCH(1,1,1) Models %%

Model_Volatility_Estimates = [GJR_PortfolioVolatility_t GJR_DCC_PortfolioVolatility_t ];
figure 
hold on;
grid on;
plot(xaxis,Log_Portfolio_Returns,'LineStyle','-','LineWidth',1,'Color','black');
for i = 1:2
    plot(xaxis,2*Model_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color',char(Colours(i)));
    plot(xaxis,-2*Model_Volatility_Estimates(:,i),'LineStyle','-','LineWidth',1,'Color',char(Colours(i)));
end
legend({'Portfolio Log Returns','2*GJR-GARCH','-2*GJR-GARCH','2*DCC-GJR-GARCH','-2*DCC-GJR-GARCH'},'Location','best');
ylim([min(Log_Portfolio_Returns)-0.01,max(Log_Portfolio_Returns)+0.01]);
title('Comparing Multivariate and Univariate Conditional Portfolio Volatilility Estimates');
xlabel('Dates');
ylabel('Portfolio Volatility');

%% Calculating VaR_5%,VaR_1%, ES_5% and ES_1% Using GJR-GARCH(1,1,1) & DCC-GJR-GARCH Model Estimates %%

% Computing Quantiles For Standard Normal Distribution %
StandardNormal1PctQuantile = norminv(0.01);
StandardNormal5PctQuantile = norminv(0.05);

% Computing VaR %
GJR_VaR_1     = -StandardNormal1PctQuantile * GJR_PortfolioVolatility_t;
GJR_VaR_5     = -StandardNormal5PctQuantile * GJR_PortfolioVolatility_t;
DCC_GJR_VaR_1 = -StandardNormal1PctQuantile * GJR_DCC_PortfolioVolatility_t;
DCC_GJR_VaR_5 = -StandardNormal5PctQuantile * GJR_DCC_PortfolioVolatility_t;

% Computing ES Factors %
n                       = 10^6;
RandomNumbers           = rand(n,1);
StandardNormalES1Factor = -mean(norminv(0.01 * RandomNumbers));
StandardNormalES5Factor = -mean(norminv(0.05 * RandomNumbers));

% Computing ES %
GJR_ES_1     = StandardNormalES1Factor * GJR_PortfolioVolatility_t;
GJR_ES_5     = StandardNormalES5Factor * GJR_PortfolioVolatility_t;
DCC_GJR_ES_1 = StandardNormalES1Factor * GJR_DCC_PortfolioVolatility_t;
DCC_GJR_ES_5 = StandardNormalES5Factor * GJR_DCC_PortfolioVolatility_t;

% Charting Results %
GJR_ChartData = [GJR_VaR_1 GJR_ES_1 GJR_VaR_5 GJR_ES_5];
DCC_GJR_ChartData = [DCC_GJR_VaR_1 DCC_GJR_ES_1 DCC_GJR_VaR_5 DCC_GJR_ES_5];
LineStyles = {'-','--'};
Colours3 = {'blue','blue','red','red'};
for i = 1:8
    if i <= 4
        ChartingData = GJR_ChartData;
        j = i;
        if i == 1
            figure
            hold on;
            grid on;
            title('GJR-GARCH(1,1,1) Value-at-Risk and Expected Shortfall Estimates For Portfolio Returns');
            xlabel('Dates');
            ylabel('Percentage');
        end
    else
        ChartData = DCC_GJR_ChartData;
        j = i - 4;
        if i == 5
            legend({'VaR_1_%','ES_1_%','VaR_5_%','ES_5_%'},'Location','best');
            figure
            hold on;
            grid on; 
            title('DCC-GJR-GARCH(1,1,1) Value-at-Risk and Expected Shortfall Estimates For Portfolio Returns');
            xlabel('Dates');
            ylabel('Percentage');
        end
    end
    k = mod(j,2) + 1;
    plot(xaxis,ChartingData(:,j),'LineStyle',char(LineStyles(k)),'LineWidth',1,'Color',char(Colours3(j)));
    if i == 8 
        legend({'VaR_1_%','ES_1_%','VaR_5_%','ES_5_%'},'Location','best');
    end
end

%% Backtesting VaR and ES %%

% Computing Breach Sequences For VaR_5% %
GJR_BreachSequence     = Log_Portfolio_Returns <= -GJR_VaR_5;
DCC_GJR_BreachSequence = Log_Portfolio_Returns <= -DCC_GJR_VaR_5;
p                      = 0.05;

% Performing Conditional and Unconditional Coverage Tests %
BreachSequences = [GJR_BreachSequence DCC_GJR_BreachSequence];
TestData        = NaN(6,2);
pData           = NaN(2,2);
pData(1,:)      = [0.05 0.05];
for i = 1:2
    % Unconditional Coverage Test %
    V1                 = sum(BreachSequences(:,i));
    V0                 = NDates - V1; 
    p_hat              = V1/NDates;
    pData(2,i)         = p_hat;
    Constrained_LogL   = V1 * log(p) + V0 * log(1-p);
    Unconstrained_LogL = V1 * log(p_hat) + V0 * log(1-p_hat);
    TestStat           = -2*(Constrained_LogL - Unconstrained_LogL);
    P_Value            = 1 - chi2cdf(TestStat,1);
    TestData(1,i)      = TestStat;
    TestData(2,i)      = P_Value;
    
    % Conditional Coverage Test %
    Aux01             = (ones(NDates - 1,1) - BreachSequences(1:(end-1),i))...
                        .* BreachSequences(2:end,i);
    p01               = sum(Aux01) / (NDates - 1 - sum(BreachSequences(1:(end-1),i)));
    Aux11             = BreachSequences(1:(end-1),i) .* BreachSequences(2:end,i);
    p11               = sum(Aux11) / sum(BreachSequences(1:(end-1),i));
    TestData(3,i)     = p01;
    TestData(4,i)     = p11;
    Constrained_LogL2 = Unconstrained_LogL; % Constrained LogLikelihood For 
                                            % Conditional Coverage Test equals
                                            % The Unconstrained
                                            % LogLikelihood For The
                                            % Unconditional Coverage Test.
    Unconstrained_LogL2 = BreachSequences(1,i) * log(p_hat) + ...
                          (1 - BreachSequences(1,i)) * log(1-p_hat); 
                          % Initialising Loop
   for j = 2:NDates
       if (BreachSequences(j-1,i) == 0 && BreachSequences(j,i) == 0)
           Unconstrained_LogL2 = Unconstrained_LogL2 + log(1 - p01);
       elseif (BreachSequences(j-1,i) == 0 && BreachSequences(j,i) == 1)
           Unconstrained_LogL2 = Unconstrained_LogL2 + log(p01);
       elseif (BreachSequences(j-1,i) == 1 && BreachSequences(j,i) == 0)
           Unconstrained_LogL2 = Unconstrained_LogL2 + log(1 - p11);
       elseif (BreachSequences(j-1,i) == 1 && BreachSequences(j,i) == 1)
           Unconstrained_LogL2 = Unconstrained_LogL2 + log(p11);        
       end
   end
   TestStat2     = -2*(Constrained_LogL2 - Unconstrained_LogL2);
   P_Value2      = 1 - chi2cdf(TestStat2,1);
   TestData(5,i) = TestStat2;
   TestData(6,i) = P_Value2;
end

% Producing a Display For Our Test Data %
CellData               = [pData; TestData];
RowHeaders             = {'Prob(P/L < -VaR) = P = ','Observed P','Unconditional Coverage Test Statistic','Unconditonal Coverage Test P-Value','Observed p01','Observed p11','Conditional Coverage Test Statistic','Conditonal Coverage Test P-Value'};
ColHeaders             = {'Models','GJR-GARCH(1,1,1)','DCC-GJR-GARCH(1,1,1)'};
DisplayData            = [RowHeaders' num2cell(CellData)];
Test_Statistic_Display = [ColHeaders; DisplayData];

% Looking at the unconditional coverage test results, we see that both models 
% are too conservative, as we are forced to reject H0: P = 0.05, with both
% models yieling a p_hat = 0.0385.

% We do note however, that when looking at the conditonal coverage test
% results, we see that we must reject H0: violations are IID over time, for
% the DCC-GJR-GARCH model at the 5% level, and conclude that there is a
% tendency for violations to cluster (in fact we have observed p11 > p01 
% for this model). In contrast, We do not reject H0 at the 5% level for
% univariate GJR-GARCH(1,1,1), (with p11 ~= p01 for this model), showing  
% there is not a tendency for violation clustering, and thus it is a  
% superior model incomparison to DCC-GJR-GARCH.
% It is therefore reasonable to conclude that this is the superior model.
